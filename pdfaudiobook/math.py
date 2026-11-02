"""Find equations on a page, recognize them as LaTeX, and turn LaTeX into speech.

Why equations need their own path: TeX-produced PDFs set math in Computer
Modern math fonts (CMMI, CMSY, CMEX) whose built-in encodings put symbols at
code points that decode as unrelated characters. A minus sign comes out as
"¡" and a plus-or-minus as "§". Radicals and fraction bars are drawn as
graphics or glyph pieces with no text at all. Tesseract is trained on text
lines, not 2-D layouts, so it does no better on scans.

Pipeline:
1. Detect math lines: spans in math fonts (text layer) or low-confidence,
   symbol-heavy lines (OCR).
2. Group neighboring math lines into one region (a fraction is three lines:
   numerator, bar, denominator), crop it, and recognize it with pix2tex
   (LaTeX-OCR, Blecher: ViT encoder + Transformer decoder; README reports
   BLEU 0.88 on its test set).
3. Speak LaTeX with the Speech Rule Engine (Sorge et al., W4A 2014; the math
   speech used by MathJax and ChromeVox) in ClearSpeak style, falling back
   to a built-in verbalizer when Node.js is not installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

MATH_FONT_RE = re.compile(
    r"CMMI|CMSY|CMEX|CMBSY|MSBM|MSAM|EUFM|EUSM|RSFS|STIX|XITS|Math|Symbol|MTMI|MTSY|Euclid|LMMono\w*Math|Asana|Cambria",
    re.IGNORECASE,
)
MATH_CHARS = set("=+−-×÷/^√∑∏∫∂∞±≤≥≠≈∈∉⊂⊆∪∩→←⇒⇔∀∃∇·′″αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ()[]{}|<>")
DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_RE = re.compile(r"(?<!\$)\$([^$]+?)\$(?!\$)")


# ------------------------------------------------------------ detection


def span_is_math(font: str, text: str) -> bool:
    return bool(MATH_FONT_RE.search(font)) and bool(text.strip())


def ocr_line_is_math(text: str, conf: float) -> bool:
    """Heuristic for Tesseract lines: math is symbol-dense and Tesseract is
    unsure of it. Prose lines rarely contain '=' and score high confidence."""
    t = text.replace(" ", "")
    if not t:
        return False
    sym = sum(1 for c in t if c in MATH_CHARS) / len(t)
    words = re.findall(r"[A-Za-z]{4,}", text)
    return ("=" in t and len(words) <= 3) or (sym > 0.25 and conf < 80) or (conf < 45 and len(words) <= 1)


def group_regions(lines, is_math) -> list[list[int]]:
    """Indices of consecutive math lines that sit close together vertically.

    Lines are close when the gap between them is under 1.2 x line height,
    which keeps a fraction's numerator and denominator in one region while
    splitting two separate display equations.
    """
    groups: list[list[int]] = []
    for i, l in enumerate(lines):
        if not is_math(l):
            continue
        if groups:
            last = lines[groups[-1][-1]]
            h = max(last.bbox[3] - last.bbox[1], 1)
            if groups[-1][-1] == i - 1 and l.bbox[1] - last.bbox[3] < 1.2 * h:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def ink_regions(page, dpi: int = 100) -> list[tuple]:
    """Bounding boxes (PDF points) of clumps of ink on a rendered page.

    Dilating the binarized page merges glyphs into word/line blobs, the
    classic run-length smoothing idea for page segmentation (Wong, Casey &
    Wahl, IBM J. Res. Dev. 1982). A fraction's numerator, bar and
    denominator merge into one blob because they are stacked closely.
    """
    import cv2
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = (img < 160).astype(np.uint8)
    k = max(2, dpi // 25)
    blob = cv2.dilate(ink, np.ones((k, 3 * k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blob)
    s = 72 / dpi
    return [
        (x * s, y * s, (x + w) * s, (y + h) * s)
        for x, y, w, h, area in stats[1:]
        if w * s > 4 and h * s > 4
    ]


def _overlap(a, b) -> float:
    """Fraction of box a covered by box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0, w) * max(0, h) / area if area > 0 else 0


def unexplained_ink(page, prose_boxes) -> list[tuple]:
    """Ink blobs that no confidently read prose line covers: on a scanned
    page these are equations, figures, or decorations."""
    # Blobs are dilated, so grow the text boxes by the same margin
    # (3k x k pixels at 100 dpi is about 9 x 3 points) before comparing.
    grown = [(b[0] - 9, b[1] - 3, b[2] + 9, b[3] + 3) for b in prose_boxes]
    H = page.rect.height
    out = []
    for r in ink_regions(page):
        if r[1] < 0.06 * H or r[3] > 0.94 * H or r[2] - r[0] < 15:
            continue  # page numbers, running heads, specks
        if sum(_overlap(r, b) for b in grown) < 0.5:
            out.append(r)
    return out


# ------------------------------------------------------------ recognition


class MathOCR:
    """pix2tex wrapper. Loaded lazily; weights download on first use."""

    _model = None
    disabled = False  # set by --fast-math

    @classmethod
    def available(cls) -> bool:
        try:
            import pix2tex  # noqa: F401

            return True
        except ImportError:
            return False

    SCALES = (100, 125, 150, 200)

    @classmethod
    def recognize(cls, page, rect) -> str:
        """Recognize at several render scales and keep the consensus.

        pix2tex output changes with image scale (on our samples a misread "="
        at one DPI was read correctly at another). Test-time augmentation
        with a medoid vote, i.e. picking the candidate most similar to all
        the others, cancels out scale-specific errors.
        """
        if cls._model is None:
            import torch
            from pix2tex.cli import LatexOCR

            # Half the cores: all of them freezes a desktop PC during a long import.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            cls._model = LatexOCR()
        # Tall regions (matrices, multi-line) take far longer per pass; one
        # mid-scale pass keeps a 700-page book within reason.
        scales = cls.SCALES if rect.height < 40 else (125,)
        cands = [cls._model(crop_region(page, rect, dpi)).strip() for dpi in scales]
        cands = [c for c in cands if looks_like_latex(c)]
        if not cands:
            return ""
        if len(cands) == 1:
            return cands[0]
        from difflib import SequenceMatcher

        def score(c):
            agree = sum(SequenceMatcher(None, c, o).ratio() for o in cands if o is not c)
            return agree - 0.5 * len(_SUSPECT_RE.findall(c))

        return max(cands, key=score)


# Tokens pix2tex emits when it misreads print (arrows for "=", stray accents
# and calligraphic letters for plain letters, empty groups).
_SUSPECT_RE = re.compile(r"\\(longrightarrow|longleftarrow|mathcal|dot|bar|ddot|tilde|jmath|imath|i\b)|\{\}|^-\{")


# Tokens that pix2tex produces when it is shown a diagram instead of a
# formula (seen on Craig's figures: \stackrel{\cal D}{\P}_{\S}, \star,
# \oint, empty groups). Printed formulas almost never use them.
_NONSENSE_RE = re.compile(r"\\(stackrel|P|S|star|phantom|vphantom|oint|widehat|overbrace|underbrace|not|cal)\b"
                          r"|\\!|\{\}|\\mathrm\{\}")


def looks_like_latex(tex: str) -> bool:
    """Reject recognizer output that is empty or just repeated junk."""
    if not tex or len(tex) > 600:
        return False
    if re.search(r"(.{3,})\1{4,}", tex):  # decoder stuck in a loop
        return False
    if len(_NONSENSE_RE.findall(tex)) >= 4:  # a diagram decoded as "math"
        return False
    # Signs of a table or figure read by the image model: braces under or
    # over numbers, arrays, and "10 to the degree" (it reads small exponent
    # digits it can't see as \circ).
    if re.search(r"\\(under|over)brace|\\begin\{array\}", tex) or tex.count("^{\\circ}") >= 2:
        return False
    # Chained arrows between stacked numbers: a table column read as math.
    if tex.count("\\longrightarrow") >= 2 or re.search(r"10\^\{[.\\]", tex):
        return False
    return bool(re.search(r"[=+\-^_\\]|\d", tex))


def crop_region(page, rect, dpi: int = 200):
    """Render a PDF rectangle to a PIL image with a small white margin."""
    import pymupdf
    from PIL import Image

    pad = 4
    r = pymupdf.Rect(rect) + (-pad, -pad, pad, pad)
    pix = page.get_pixmap(dpi=dpi, clip=r & page.rect, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB")


# ------------------------------------------------------------ speech


_NODE_DIR = os.path.join(os.path.expanduser("~"), ".pdfaudiobook", "node")
_SRE_JS = os.path.join(os.path.dirname(__file__), "sre_server.mjs")


# A subscripted letter followed by another letter, as in x_{0}e^{-at}, is
# a product, but SRE's semantic parser reads it as function application
# ("x sub 0 of e"). Bracing the subscripted symbol, {x_{0}}e, makes it one
# operand. Big operators (sum, integral) and a following "(" are left alone,
# since x_{p}(t) really is a function of t.
_SUBSCRIPTED_RE = re.compile(r"(?<![\\a-zA-Z])([A-Za-z]|\\(?!sum|int|prod|lim|min|max|oint)[a-zA-Z]+)"
                             r"_(\{[^{}]*\}|[A-Za-z0-9])(?=\s*(?:[A-Za-z]|\\[a-zA-Z]))")


def _group_subscripts(tex: str) -> str:
    def fix(m):
        base = m.group(1)
        # If the same letter is used as a function elsewhere, x(t) = x_{0}e^{..},
        # SRE keeps reading x_{0} as one; only an explicit product sign stops it.
        if re.search(re.escape(base) + r"\s*\(", tex):
            return f"{base}_{m.group(2)}\\cdot "
        return f"{{{base}_{m.group(2)}}}"
    return _SUBSCRIPTED_RE.sub(fix, tex)


# A subscript of three or more letters is a word label (I_ave, Q_electron,
# V_out), which SRE would spell out letter by letter ("sub a v e"); as
# \text{} it is read as one word ("sub ave").
_WORD_SUB_RE = re.compile(r"_\{\s*([A-Za-z](?:\s*[A-Za-z]){2,})\s*\}")


def _word_subscripts(tex: str) -> str:
    return _WORD_SUB_RE.sub(lambda m: "_{\\text{" + re.sub(r"\s+", "", m.group(1)) + "}}", tex)


# Words the book prints between two equations on one line ("V = U/q  or
# dV = dU/q") come out of the glyph rebuild glued to the next symbol
# ("orV_{B}"), and SRE reads them as variables o times r. Mark them as text.
_CONNECTIVE_RE = re.compile(r"(?<![A-Za-z\\])\s*(or|and)\s*(?=[A-Z\\(∆Δ])")


def _connectives(tex: str) -> str:
    tex = _CONNECTIVE_RE.sub(lambda m: "\\text{ " + m.group(1) + " }", tex)
    # Image-model styling: \mathrm makes SRE say "normal I", and a unit word
    # in \mathrm is spelled letter by letter; keep the letter, read the word.
    tex = re.sub(r"\\(textstyle|displaystyle)\b", "", tex)
    tex = re.sub(r"\\operatorname\*?\{(lim|sin|cos|tan|log|ln|exp)\}", r"\\\1", tex)
    tex = re.sub(r"\\mathrm\{\\Delta\}", "∆", tex)
    tex = re.sub(r"\\math(?:rm|sf)\{\s*\{?([A-Za-z]{2,}[A-Za-z/]*)\}?\s*\}", r"\\text{\1}", tex)
    tex = re.sub(r"\\math(rm|sf)\{", "{", tex)
    # SRE says "lim over t right arrow 0 of"; say it the way a lecturer does.
    tex = re.sub(r"\\lim_\{(.+?)\\to\s*(.+?)\}",
                 r"\\text{the limit as }\1\\text{ approaches }\2\\text{ of }", tex)
    # SRE reads a capital delta as "triangle" (or "increment" for U+2206);
    # in a physics text it is always "delta", as in delta Q over delta t.
    return re.sub(r"\\Delta(?![A-Za-z])|[∆Δ]", r"\\text{delta}\\,", tex)


class MathSpeaker:
    """LaTeX to English. Uses SRE through one long-lived Node process."""

    def __init__(self, style: str = "clearspeak"):
        self.style = style
        self.proc = None
        node = shutil.which("node")
        if node and self._ensure_packages():
            env = dict(os.environ, NODE_PATH=os.path.join(_NODE_DIR, "node_modules"))
            self.proc = subprocess.Popen(
                [node, _SRE_JS, style], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, env=env, cwd=_NODE_DIR,
            )
            if self.proc.stdout.readline().strip() != "ready":
                self.proc = None

    @staticmethod
    def _ensure_packages() -> bool:
        if os.path.isdir(os.path.join(_NODE_DIR, "node_modules", "speech-rule-engine")):
            return True
        npm = shutil.which("npm")
        if not npm:
            return False
        os.makedirs(_NODE_DIR, exist_ok=True)
        r = subprocess.run(
            [npm, "install", "--silent", "--prefix", _NODE_DIR, "speech-rule-engine@4", "mathjax-full@3"],
            capture_output=True,
        )
        return r.returncode == 0

    @property
    def engine(self) -> str:
        return "sre" if self.proc else "builtin"

    def speak(self, tex: str) -> str:
        if self.proc:
            try:
                self.proc.stdin.write(json.dumps(_group_subscripts(_word_subscripts(_connectives(tex)))) + "\n")
                self.proc.stdin.flush()
                out = json.loads(self.proc.stdout.readline())
                if out:
                    return out
            except (OSError, ValueError):
                self.proc = None
        return latex_to_words(tex)

    def speak_text(self, text: str) -> str:
        """Replace every $$...$$ in a paragraph with its spoken form."""
        text = DISPLAY_RE.sub(lambda m: f" {self.speak(m.group(1))}. ", text)
        text = INLINE_RE.sub(lambda m: self.speak(m.group(1)), text)
        # SRE says "x dot above" for \dot{x}; engineers say "x dot".
        text = text.replace(" dot above", " dot").replace(" double dot above", " double dot")
        text = text.replace("[equation]", " An equation appears here. ")
        text = text.replace("[table]", " A table appears here. ")
        return re.sub(r"[ \t]+", " ", text).replace(" .", ".")


_ARROWS_RE = re.compile(r"\\(long)?(right|left|leftright)arrow\b|\\to\b")


def reconcile_relations(tex: str, printed: str) -> str:
    """Correct the image model's relation symbols using the text layer.

    pix2tex sometimes reads a wide "=" as an arrow ("x long right arrow ...").
    When the page's own glyphs for the formula hold "=" and no arrow, every
    arrow in the LaTeX is that "=".
    """
    if "=" in printed and not any(a in printed for a in "→←↔⟶⟵⇒⇐⇔"):
        tex = _ARROWS_RE.sub("=", tex)
    # It also reads a printed increment sign ∆ as a capital lambda.
    if any(c in printed for c in "∆Δ") and "Λ" not in printed:
        tex = tex.replace("\\Lambda", "\\Delta")
    return tex


# Built-in verbalizer: covers the common school/undergrad constructs so math
# is still spoken sensibly without Node.js. SRE is better and preferred.

_GREEK = "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega".split()
_SYMBOLS = {
    "pm": "plus or minus", "mp": "minus or plus", "times": "times", "cdot": "times", "div": "divided by",
    "leq": "is less than or equal to", "le": "is less than or equal to", "geq": "is greater than or equal to",
    "ge": "is greater than or equal to", "neq": "is not equal to", "ne": "is not equal to", "approx": "is approximately",
    "equiv": "is equivalent to", "sim": "is similar to", "propto": "is proportional to", "infty": "infinity",
    "partial": "partial", "nabla": "del", "in": "in", "notin": "not in", "subset": "subset of",
    "subseteq": "subset of or equal to", "cup": "union", "cap": "intersection", "to": "to", "rightarrow": "goes to",
    "Rightarrow": "implies", "iff": "if and only if", "forall": "for all", "exists": "there exists",
    "ldots": "dot dot dot", "cdots": "dot dot dot", "dots": "dot dot dot", "prime": "prime", "circ": "degrees",
    "log": "log", "ln": "natural log", "sin": "sine", "cos": "cosine", "tan": "tangent", "exp": "exp",
    "lim": "the limit", "max": "max", "min": "min", "det": "determinant", "angle": "angle", "deg": "degrees",
}
_BIG = {"sum": "the sum", "prod": "the product", "int": "the integral", "oint": "the contour integral", "lim": "the limit"}
_CHARS = {"=": " equals ", "+": " plus ", "-": " minus ", "−": " minus ", "<": " is less than ", ">": " is greater than ",
          "/": " over ", "!": " factorial ", ",": ", ", "(": " open paren ", ")": " close paren ", "|": " absolute value ",
          "[": " open bracket ", "]": " close bracket "}
_POWERS = {"2": "squared", "3": "cubed"}
_TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|[{}^_]|\d+(?:\.\d+)?|\s+|.")
_IGNORE = {"left", "right", "big", "Big", "bigg", "Bigg", "displaystyle", "mathrm", "mathbf", "mathit", "operatorname",
           "text", "mathbb", "mathcal", ",", ";", ":", "!", " ", "quad", "qquad", "limits", "nolimits"}


def latex_to_words(tex: str) -> str:
    # Arrays and matrices: read row by row.
    tex = re.sub(r"\\begin\{(array|[pbvB]?matrix|cases|aligned)\}(\{[^}]*\})?", " ", tex)
    tex = re.sub(r"\\end\{(array|[pbvB]?matrix|cases|aligned)\}", " ", tex)
    tex = tex.replace("&", " , ").replace("\\\\", " ; next row, ")
    toks = [t for t in _TOKEN_RE.findall(tex) if not t.isspace()]
    pos = 0

    def group() -> str:
        nonlocal pos
        if pos >= len(toks):
            return ""
        if toks[pos] == "{":
            pos += 1
            out = seq("}")
            pos += 1
            return out
        return atom()

    def raw_group() -> str:
        """Group content for exponent tests like ^{2}."""
        if pos < len(toks) and toks[pos] == "{" and pos + 2 < len(toks) and toks[pos + 2] == "}":
            return toks[pos + 1]
        return toks[pos] if pos < len(toks) else ""

    def atom() -> str:
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t.startswith("\\"):
            name = t[1:]
            if name == "frac" or name == "dfrac" or name == "tfrac":
                a, b = group(), group()
                return f" the fraction {a} over {b}, "
            if name == "sqrt":
                idx = ""
                if pos < len(toks) and toks[pos] == "[":
                    pos += 1
                    idx = seq("]")
                    pos += 1
                body = group()
                return f" the {idx + ' th' if idx else 'square'} root of {body}, "
            if name in _BIG:
                lo = hi = ""
                while pos < len(toks) and toks[pos] in "_^":
                    which = toks[pos]
                    pos += 1
                    g = group()
                    lo, hi = (g, hi) if which == "_" else (lo, g)
                rng = f" from {lo}" if lo else ""
                rng += f" to {hi}" if hi else ""
                return f" {_BIG[name]}{rng} of "
            if name in _GREEK or name.lower() in _GREEK:
                return f" {('capital ' if name[0].isupper() else '')}{name.lower().replace('var', '')} "
            if name in ("text", "mathrm", "operatorname", "mathbf", "mathit", "mathbb", "mathcal"):
                return " " + group() + " "
            if name in _SYMBOLS:
                return f" {_SYMBOLS[name]} "
            if name in _IGNORE:
                return " "
            return f" {name} "
        if t == "^":
            r = raw_group()
            if r in _POWERS:
                group()
                return f" {_POWERS[r]} "
            return f" to the power {group()}, "
        if t == "_":
            return f" sub {group()} "
        if t in _CHARS:
            return _CHARS[t]
        return f" {t} "

    def seq(end: str | None = None) -> str:
        nonlocal pos
        out = []
        while pos < len(toks) and toks[pos] != end:
            if toks[pos] == "}":  # unbalanced; skip
                pos += 1
                continue
            out.append(group())
        return "".join(out)

    words = seq()
    words = re.sub(r"\s+", " ", words)
    words = re.sub(r"\s+,", ",", words)
    return re.sub(r",\s*,", ",", words).strip(" ,")
