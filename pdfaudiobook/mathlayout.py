"""Rebuild math as LaTeX from the PDF's own glyph positions.

When a PDF's math fonts decode to correct Unicode (most publisher PDFs
since the 2000s, e.g. Springer, Wiley), every glyph is already known
exactly. What text extraction loses is the 2-D layout: e^{-αt} comes out as
"e−αt" and the exponent is lost. The PDF still records each glyph's
baseline and size, so superscripts (smaller, raised), subscripts (smaller,
lowered), accents (a dot glyph over a letter) and fractions (glyphs above
and below a horizontal rule) can be recovered by geometry alone. This is
the approach of Baker, Sexton & Sorge, "A Linear Grammar Approach to
Mathematical Formula Recognition from PDF" (MKM 2009), simplified. It is
exact where it applies and about 1000x faster than running an image model.

Glyphs that decode as junk (TeX fonts without a usable encoding) cannot be
rebuilt this way; those regions go to image-based math OCR instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .math import MATH_FONT_RE

ACCENTS = {"\u0307": r"\dot", "\u0308": r"\ddot", "\u0302": r"\hat", "\u0303": r"\tilde", "\u0304": r"\bar",
           "\u20d7": r"\vec", "˙": r"\dot", "¨": r"\ddot", "ˆ": r"\hat", "^": r"\hat", "˜": r"\tilde", "~": r"\tilde",
           "¯": r"\bar", "→": r"\vec", "⃗": r"\vec", "´": r"\acute", "`": r"\grave"}
OPS = set("=+−-×÷·<>≤≥≠≈≡∼∝±∓∈∉⊂⊆∪∩→←↔⇒⇔∀∃∂∇∑∏∫∮√∞/|()[]{},.;:!′″")
EQ_NUM_RE = re.compile(r"^\(\d+(\.\d+)*[a-z]?\)$")


@dataclass
class Glyph:
    c: str
    x0: float
    y0: float
    x1: float
    y1: float
    oy: float  # baseline y
    size: float
    font: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


def glyphs_of_line(line: dict) -> list[Glyph]:
    out = []
    for s in line["spans"]:
        for ch in s["chars"]:
            x0, y0, x1, y1 = ch["bbox"]
            oy = ch["origin"][1]
            if re.search(r"CMEX|EXTENSION", s["font"], re.I):
                # TeX extension-font glyphs (big brackets, display integrals)
                # hang down from their origin; their visual baseline is about
                # 0.85 em lower.
                oy += 0.85 * s["size"]
            out.append(Glyph(ch["c"], x0, y0, x1, y1, oy, s["size"], s["font"]))
    return out


def is_junk(g: Glyph) -> bool:
    """A math-font glyph whose Unicode is clearly wrong (built-in TeX font
    encodings decode a minus as U+00A1 and so on) or missing."""
    o = ord(g.c)
    if g.c == "�" or 0xE000 <= o <= 0xF8FF or o < 32:
        return True
    return bool(MATH_FONT_RE.search(g.font)) and 0xA1 <= o <= 0xBF and g.c not in "±·¬°"


def _is_italic(font: str) -> bool:
    return bool(re.search(r"Italic|Oblique|CMMI|CMTI|-It\b|MathItalic", font, re.I))


def _is_mathfont(font: str) -> bool:
    return bool(MATH_FONT_RE.search(font)) or bool(re.match(r"CMR\d", font, re.I))


def _baseline(glyphs: list[Glyph]) -> tuple[float, float]:
    """Dominant (baseline, size): the most common baseline among the
    largest glyphs, which are the ones on the main line of the formula."""
    big = max(g.size for g in glyphs)
    main = [g for g in glyphs if g.size >= 0.9 * big] or glyphs
    ys = sorted(round(g.oy, 0) for g in main)
    return ys[len(ys) // 2], big


FUNCS = {"sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
         "log", "ln", "exp", "lim", "max", "min", "sup", "inf", "det", "deg", "dim", "ker", "arg", "gcd"}


def _level(g: Glyph, base: float, size: float) -> str:
    if g.size < 0.88 * size:
        if g.oy < base - 0.15 * size:
            return "^"
        if g.oy > base + 0.08 * size:
            return "_"
    return ""


def linear_latex(glyphs: list[Glyph]) -> str:
    """One row of math with scripts (nested, by recursion) and accents."""
    glyphs = [g for g in sorted(glyphs, key=lambda g: g.x0) if not g.c.isspace()]
    if not glyphs:
        return ""
    base, size = _baseline(glyphs)
    out: list[str] = []
    i = 0
    while i < len(glyphs):
        g = glyphs[i]
        lv = _level(g, base, size)
        if lv:
            # Collect the whole script run; it may contain its own scripts.
            j = i
            while j < len(glyphs) and _level(glyphs[j], base, size) == lv:
                j += 1
            out.append(f"{lv}{{{linear_latex(glyphs[i:j])}}}")
            i = j
            continue
        if g.c in ACCENTS:
            # Accent glyphs sit above a letter; the PDF may list them before
            # or after it.
            # Glyph boxes span the full font height, so test horizontal
            # overlap only, and attach to whichever neighbor is closer.
            dp = abs(glyphs[i - 1].xc - g.xc) if i > 0 and out else 1e9
            dn = abs(glyphs[i + 1].xc - g.xc) if i + 1 < len(glyphs) else 1e9
            prev_ok = dp < 0.5 * size and dp <= dn
            next_ok = dn < 0.5 * size and dn < dp
            if g.x1 - g.x0 < 0.05 and i + 1 < len(glyphs):
                # Zero-width combining mark: Unicode puts it after its base
                # letter, but some typesetters emit it before. Attach to the
                # previous glyph only if that is a letter.
                prev_letter = i > 0 and out and glyphs[i - 1].c.isalnum()
                prev_ok, next_ok = bool(prev_letter), not prev_letter
            if prev_ok:
                out[-1] = f"{ACCENTS[g.c]}{{{out[-1]}}}"
                i += 1
                continue
            if next_ok:
                out.append(f"{ACCENTS[g.c]}{{{_tex_char(glyphs[i + 1].c)}}}")
                i += 2
                continue
        if g.c.isalpha() and g.c.isascii():
            j = i
            while j < len(glyphs) and glyphs[j].c.isalpha() and glyphs[j].c.isascii() and not _level(glyphs[j], base, size) \
                    and _is_italic(glyphs[j].font) == _is_italic(g.font):
                j += 1
            word = "".join(x.c for x in glyphs[i:j])
            if word in FUNCS:
                out.append(f"\\{word} ")
                i = j
                continue
        out.append(_tex_char(g.c))
        i += 1
    tex = "".join(out).replace("···", r"\cdots ").replace("...", r"\ldots ")
    return re.sub(r"\s+", " ", tex).strip()


def _tex_char(c: str) -> str:
    return {"…": r"\ldots ", "⋯": r"\cdots ", "{": r"\{", "}": r"\}", "%": r"\%", "#": r"\#", "&": r"\&", "$": r"\$",
            "\\": r"\backslash "}.get(c, c)


def region_latex(glyphs: list[Glyph], bars: list[tuple]) -> str:
    """A display region: rows of math, fraction rules, and radicals.

    `bars` are thin horizontal rules (x0, y, x1) from the page drawings.
    Each bar with glyphs above and below it becomes \\frac{above}{below}.
    A bar starting at the right edge of a √ glyph is a radical's overline.
    """
    glyphs = [g for g in glyphs if not g.c.isspace()]
    if not glyphs:
        return ""
    tokens: list[tuple[float, float, str]] = []  # (x0, baseline, latex)
    used: set[int] = set()
    # Big operators with limits set above and below them.
    big_size = max(g.size for g in glyphs)
    for k, op in enumerate(glyphs):
        if op.c not in BIG_OPS:
            continue
        lo_lim = [j for j, g in enumerate(glyphs) if j != k and j not in used and op.x0 - 2 <= g.xc <= op.x1 + 2
                  and g.size < 0.88 * big_size and g.oy > op.oy + 0.25 * op.size]
        hi_lim = [j for j, g in enumerate(glyphs) if j != k and j not in used and op.x0 - 2 <= g.xc <= op.x1 + 2
                  and g.size < 0.88 * big_size and g.oy < op.oy - 0.9 * op.size]
        if lo_lim or hi_lim:
            used.update(lo_lim + hi_lim + [k])
            tex = {"∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint", "⋃": r"\bigcup", "⋂": r"\bigcap"}[op.c]
            if lo_lim:
                tex += "_{" + linear_latex([glyphs[j] for j in lo_lim]) + "}"
            if hi_lim:
                tex += "^{" + linear_latex([glyphs[j] for j in hi_lim]) + "}"
            tokens.append((op.x0, op.oy, tex))
    # Radicals first: the overline sits at the top of the √ glyph.
    for bx0, by, bx1 in sorted(bars, key=lambda b: b[2] - b[0]):
        rad = next((k for k, g in enumerate(glyphs) if k not in used and g.c == "√"
                    and abs(g.x1 - bx0) < 2.5 and abs(g.y0 - by) < 3), None)
        if rad is not None:
            inner = [k for k, g in enumerate(glyphs) if k not in used and k != rad
                     and bx0 - 1 <= g.xc <= bx1 + 1 and by < g.oy < by + 2.2 * g.size]
            used.update(inner + [rad])
            tex = r"\sqrt{" + linear_latex([glyphs[k] for k in inner]) + "}"
            tokens.append((glyphs[rad].x0, glyphs[rad].oy, tex))
            continue
        # Numerator baseline sits a little above the bar, denominator's top a
        # little below it; anything further away belongs to another row.
        num = [k for k, g in enumerate(glyphs) if k not in used and bx0 - 1 <= g.xc <= bx1 + 1
               and by - 1.3 * g.size < g.oy < by + 0.5]
        den = [k for k, g in enumerate(glyphs) if k not in used and bx0 - 1 <= g.xc <= bx1 + 1
               and by < g.oy < by + 1.6 * g.size]
        if num and den:
            used.update(num + den)
            tex = r"\frac{" + linear_latex([glyphs[k] for k in num]) + "}{" + linear_latex([glyphs[k] for k in den]) + "}"
            mid = by + 0.3 * glyphs[num[0]].size
            tokens.append((bx0, mid, tex))
    rest = [g for k, g in enumerate(glyphs) if k not in used]
    rows = _rows(rest)
    if not tokens:
        return ", ".join(linear_latex(r) for r in rows)
    # The row level with the fraction bars is the formula's main line; other
    # rows are separate formulas stacked in the same display.
    mid_y = sum(t[1] for t in tokens) / len(tokens)
    main = min(rows, key=lambda r: abs(_baseline(r)[0] - mid_y)) if rows else []
    others = [r for r in rows if r is not main]
    rest = main
    # Merge fraction/radical tokens with the remaining glyphs left to right.
    # Fraction tokens are placed as pseudo-glyphs on the main baseline.
    base, size = _baseline(rest) if rest else (tokens[0][1], 10)
    pieces: list[tuple[float, str]] = [(x, t) for x, _, t in tokens]
    run: list[Glyph] = []
    for g in sorted(rest, key=lambda g: g.x0):
        nxt = next((x for x, _ in pieces if x > (run[-1].x0 if run else -1) and x < g.x0), None)
        if nxt is not None and run:
            pieces.append((run[0].x0, linear_latex(run)))
            run = []
        run.append(g)
    if run:
        pieces.append((run[0].x0, linear_latex(run)))
    body = " ".join(t for _, t in sorted(pieces, key=lambda p: p[0]))
    above = [linear_latex(r) for r in others if _baseline(r)[0] < mid_y]
    below = [linear_latex(r) for r in others if _baseline(r)[0] >= mid_y]
    return ", ".join(above + [body] + below)


BIG_DELIMS = set("⎡⎤⎢⎥⎣⎦⎧⎨⎩⎪⎫⎬⎭⎛⎞⎜⎟⎝⎠⌈⌉⌊⌋")
BIG_OPS = set("∑∏∫∮⋃⋂")


def is_complex(glyphs: list[Glyph], bars: list[tuple], body_size: float) -> bool:
    """Layouts the geometric rebuild does not handle: matrices and
    determinants, big stretched delimiters, cases, multi-row alignments
    mixed with fractions. These go to the image model instead."""
    vis = [g for g in glyphs if not g.c.isspace()]
    if any(g.c in BIG_DELIMS for g in vis):
        return True
    tall = sorted((g for g in vis if (g.y1 - g.y0) > 2.2 * body_size and g.c in "()[]{}|"), key=lambda g: g.x0)
    # A tall pair of parentheses around a fraction is still linear,
    # (a/b); anything else tall (matrix brackets, cases) is not.
    if tall and not (len(tall) == 2 and tall[0].c + tall[1].c in ("()", "[]")
                     and any(tall[0].x1 <= b[0] and b[2] <= tall[1].x0 for b in bars)):
        return True
    # Matrices and determinants: two or more rows that each have cells
    # separated by wide gaps.
    rows = [split_equation_number(r)[0] for r in _rows(vis)]
    def cells_gap(r):
        r = sorted(r, key=lambda g: g.x0)
        return any(b.x0 - a.x1 > 2.0 * body_size for a, b in zip(r, r[1:]))
    return sum(cells_gap(r) for r in rows) >= 2


def bands(glyphs: list[Glyph], bars: list[tuple]) -> list[tuple[list[Glyph], list[tuple]]]:
    """Split a display region into independent horizontal bands.

    Each glyph covers [baseline - 0.7 em, baseline + 0.2 em] and each bar
    its own height; bands are separated by vertical gaps no glyph or bar
    covers. A fraction joins its numerator, bar and denominator into one
    band, while stacked separate equations fall into separate bands.
    """
    items = [(g.oy - 0.7 * g.size, g.oy + 0.2 * g.size, "g", g) for g in glyphs if not g.c.isspace()]
    items += [(b[1] - 0.8, b[1] + 0.8, "b", b) for b in bars]
    items.sort(key=lambda t: t[0])
    out: list[tuple[list, list]] = []
    end = None
    for top, bot, kind, obj in items:
        if end is None or top > end:
            out.append(([], []))
            end = bot
        else:
            end = max(end, bot)
        (out[-1][0] if kind == "g" else out[-1][1]).append(obj)
    out = [(gs, bs) for gs, bs in out if gs]
    # Limits of a big operator (the n above a sum, the i=1 below it) sit in
    # their own thin band; fold them into the band with the operator.
    big = max((g.size for gs, _ in out for g in gs), default=0)
    k = 0
    while k < len(out):
        gs, bs = out[k]
        if all(g.size < 0.88 * big for g in gs):
            for j in (k + 1, k - 1):
                if 0 <= j < len(out) and any(o.c in BIG_OPS and any(o.x0 - 2 <= g.xc <= o.x1 + 2 for g in gs)
                                             for o in out[j][0]):
                    out[j] = (out[j][0] + gs, out[j][1] + bs)
                    out.pop(k)
                    k -= 1
                    break
        k += 1
    return out


def _rows(glyphs: list[Glyph]) -> list[list[Glyph]]:
    """Cluster glyphs into rows by baseline. Scripts stay with their row
    because they are within half a line of it."""
    if not glyphs:
        return []
    size = max(g.size for g in glyphs)
    main = sorted((g for g in glyphs if g.size >= 0.88 * size), key=lambda g: g.oy)
    bases: list[float] = []
    for g in main:
        if not bases or g.oy - bases[-1] > 0.9 * size:
            bases.append(g.oy)
    rows = [[] for _ in bases]
    for g in glyphs:
        k = min(range(len(bases)), key=lambda i: abs(bases[i] - g.oy))
        rows[k].append(g)
    return [r for r in rows if r]


def _word_is_math(word: list[Glyph]):
    """True, False, or None (neutral: digits/operators that join a math run)."""
    txt = "".join(g.c for g in word)
    big = max(x.size for x in word)
    # Letter runs on the main baseline only, so "e^{-v_max t}" is not "vmaxt",
    # and split where the font style changes, so roman "cos" + italic "t"
    # is not the word "cost".
    chars, prev = [], None
    for g in word:
        style = _is_italic(g.font)
        if prev is not None and style != prev:
            chars.append(" ")
        chars.append(g.c if g.size >= 0.85 * big else " ")
        prev = style
    runs = re.findall(r"[A-Za-z]+", "".join(chars))
    prose_run = any(len(r) >= 4 and r.lower() not in FUNCS for r in runs)
    if prose_run:
        return False  # real words, even with a footnote marker attached
    if any(g.c in "˙¨\u0307\u0308" for g in word):
        return True
    if any(g.size < 0.85 * big for g in word):
        return True  # has a script
    if any(MATH_FONT_RE.search(g.font) for g in word if not g.c.isspace()):
        return True
    letters = [g for g in word if g.c.isalpha()]
    if letters and all(len(r) == 1 for r in runs) and all(_is_italic(g.font) for g in letters):
        return True  # x, x(t), f(x,y): italic single letters are variables
    if any(r in FUNCS for r in runs) and all(r in FUNCS or len(r) == 1 for r in runs):
        return True  # cos2t, sin x
    if txt and all(c in OPS or c.isdigit() for c in txt):
        return None  # neutral: joins a math run if next to one
    if re.match(r"CMR\d", word[0].font, re.I) and not any(len(w) > 1 for w in re.findall(r"[A-Za-z]+", txt)):
        return True
    return False


def split_words(glyphs: list[Glyph]) -> list[list[Glyph]]:
    words, cur = [], []
    for g in glyphs:
        if g.c.isspace():
            if cur:
                words.append(cur)
                cur = []
            continue
        if cur and g.x0 - cur[-1].x1 > 0.22 * g.size and not g.c in ACCENTS:
            words.append(cur)
            cur = []
        cur.append(g)
    if cur:
        words.append(cur)
    return words


def flags_ok(word: list[Glyph]) -> bool:
    """Small words that can be exponents/indices: digits, single letters, signs."""
    t = "".join(g.c for g in word)
    return len(t) <= 4 and not re.search(r"[A-Za-z]{2,}", t)


def line_text(glyphs: list[Glyph]) -> tuple[str, bool]:
    """Text for one line, with inline math wrapped in $...$.

    Returns (text, is_display) where is_display means the whole line is math.
    """
    words = split_words(glyphs)
    if not words:
        return "", False
    line_size = max(g.size for w in words for g in w)
    # A word set entirely in script size (an exponent split off by spacing,
    # like the "2" of "mc 2") belongs to the math, not the prose.
    flags = [True if max(g.size for g in w) < 0.85 * line_size and flags_ok(w) else _word_is_math(w) for w in words]
    # Neutral words (digits, operators) join a math run when math is on both
    # sides, or when the math on the left ends in an operator ("t = 0").
    for i, f in enumerate(flags):
        if f is None:
            left = bool(flags[i - 1]) if i else False
            right = bool(next((x for x in flags[i + 1:] if x is not None), False))
            ops = "=+−-<>≤≥≠≈×·/("
            left_op = i > 0 and words[i - 1][-1].c in ops
            right_op = i + 1 < len(words) and (words[i + 1][0].c in ops or words[i][-1].c in ops)
            right_next = i + 1 < len(words) and bool(flags[i + 1])
            flags[i] = (left and right) or (left and left_op) or (right_next and right_op)
    # Trailing equation number like "(2.12)" is not part of the formula.
    eqnum = ""
    last = "".join(g.c for g in words[-1])
    if len(words) > 1 and EQ_NUM_RE.match(last):
        eqnum = last
        words, flags = words[:-1], flags[:-1]
    if all(flags):
        tex = linear_latex([g for w in words for g in w])
        if tex in ("⊓⊔", "□", "■", "∎"):
            return "", False  # end-of-proof mark
        tex = tex.rstrip(" .,;")
        return (f"$${tex}$$" + (f" (Equation {eqnum[1:-1]})" if eqnum else "")), True
    parts, i = [], 0
    while i < len(words):
        if flags[i]:
            j = i
            while j < len(words) and flags[j]:
                j += 1
            run = [g for w in words[i:j] for g in w]
            # Keep sentence punctuation outside the math.
            tail = ""
            while run and run[-1].c in ".,;:" and len(run) > 1:
                tail = run[-1].c + tail
                run = run[:-1]
            parts.append(f"${linear_latex(run)}${tail}")
            i = j
        else:
            # Drop footnote markers: small raised digits at the end of a word.
            w = words[i]
            big = max(g.size for g in w)
            w = [g for g in w if not (g.c.isdigit() and g.size < 0.85 * big)]
            parts.append("".join(g.c for g in w))
            i += 1
    text = " ".join(parts)
    if eqnum:
        text += f" {eqnum}"
    return text, False


# ------------------------------------------------------------ glyph names

# TeX's extension font (CMEX) holds the big delimiters and operators. Its
# glyph names (from the Computer Modern Type 1 fonts) are descriptive, e.g.
# "parenleftbigg" or "integraldisplay", so a name prefix gives the symbol.
_NAME_PREFIX = [
    ("parenleft", "("), ("parenright", ")"), ("bracketleft", "["), ("bracketright", "]"),
    ("braceleft", "{"), ("braceright", "}"), ("angleleft", "⟨"), ("angleright", "⟩"),
    ("floorleft", "⌊"), ("floorright", "⌋"), ("ceilingleft", "⌈"), ("ceilingright", "⌉"),
    ("integral", "∫"), ("contintegral", "∮"), ("summation", "∑"), ("product", "∏"), ("coproduct", "∐"),
    ("radical", "√"), ("union", "⋃"), ("intersection", "⋂"), ("circleplus", "⨁"), ("circlemultiply", "⨂"),
    ("slash", "/"), ("backslash", "\\"), ("bar", "|"), ("vextend", "|"), ("arrow", "↑"),
]


def _font_glyphs(doc, font_name: str, cache: dict):
    """(glyph order, glyph set) of an embedded font by base name, or None."""
    if font_name in cache:
        return cache[font_name]
    from .glyphmatch import glyphset

    found = None
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "Type")[1] != "/Font":
                continue
            base = doc.xref_get_key(xref, "BaseFont")[1].lstrip("/")
        except Exception:
            continue
        if base.split("+")[-1] != font_name.split("+")[-1]:
            continue
        try:
            _, ext, _, buf = doc.extract_font(xref)
            if buf and ext in ("cff", "ttf", "otf"):
                found = glyphset(buf, ext)
                break
        except Exception:
            continue
    cache[font_name] = found
    return found


def _glyph_names(doc, font_name: str, cache: dict) -> list[str] | None:
    fg = _font_glyphs(doc, font_name, cache)
    return fg[0] if fg else None


GREEK_FONT_RE = re.compile(r"Grk|Greek|Gree\b|SymbolGreek", re.I)
# Text families whose letters are what they say; everything else that uses
# Latin glyph names only for scattered single letters is checked by shape.
TEXT_FONT_RE = re.compile(r"Times|Roman|Helvet|Arial|Garamond|Palatino|Minion|Myriad|Optima|Courier|Avenir|"
                          r"Futura|Gill|Georgia|Calibri|Cambria|Verdana|Frutiger|Univers|Baskerville|Caslon|"
                          r"Century|Bookman|Sabon|Janson|Bembo|Charter|Utopia|Lucida|Segoe|CMR|CMTI|CMBX|CMSS|"
                          r"CMTT|LM(Roman|Sans|Mono)|Libertin|Source|Noto|DejaVu|Liberation|STIX|Inter", re.I)


def _symbol_like_fonts(raw: dict) -> set[str]:
    """Fonts on the page that are probably symbol fonts in disguise: not a
    known text family, and never used for a run of 3+ letters."""
    runs: dict[str, int] = {}
    for b in raw["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                f = s["font"]
                if TEXT_FONT_RE.search(f):
                    continue
                t = "".join(c["c"] for c in s["chars"])
                longest = max((len(w) for w in re.findall(r"[A-Za-z]+", t)), default=0)
                runs[f] = max(runs.get(f, 0), longest)
    return {f for f, n in runs.items() if n <= 2}


def _name_to_char(name: str) -> str | None:
    from fontTools import agl

    # The AGL maps these names to the increment, ohm and micro signs; in a
    # symbol font they are the Greek letters.
    u = {"Delta": "Δ", "Omega": "Ω", "mu": "μ"}.get(name) or agl.toUnicode(name)
    if u and u != "�":
        return u
    for prefix, ch in _NAME_PREFIX:
        if name.startswith(prefix):
            return ch
    return None


def repair_unknown_glyphs(page, raw: dict, cache: dict) -> None:
    """Fix characters whose font gives the wrong Unicode, in place.

    1. U+FFFD/private-use characters: look up the glyph's name in the
       embedded font ("integral", "parenleftbig") and map it with the Adobe
       Glyph List or a name prefix.
    2. Opaque names ("H11001") or Greek fonts that name their glyphs with
       Latin letters ("s" drawn as sigma): identify by outline shape
       (glyphmatch.py).
    """
    symbolish = _symbol_like_fonts(raw)

    def suspect(ch, font):
        o = ord(ch["c"]) if ch["c"] else 0
        if ch["c"] == "\ufffd" or 0xE000 <= o <= 0xF8FF:
            return True
        if GREEK_FONT_RE.search(font):
            return ch["c"].isascii() and ch["c"].isalpha()
        # Symbol fonts in disguise reuse Latin-1 codes: "a" for a big sigma,
        # "¢" for Delta. Only a clear shape match changes them.
        # Only letters, digits and currency signs: symbol fonts reuse those
        # codes for other symbols ("a" for a big sigma, "¢" for Delta).
        # Punctuation and math marks are left alone; TeX fonts set real
        # commas and centered dots, and shape matching confuses small marks.
        return font in symbolish and unicodedata.category(ch["c"][:1] or " ") in ("Lu", "Ll", "Nd", "Sc")

    bad = [(ch, s["font"]) for b in raw["blocks"] for l in b.get("lines", []) for s in l["spans"]
           for ch in s["chars"] if suspect(ch, s["font"])]
    if not bad:
        return
    from .glyphmatch import match_outline

    gid_at = {}
    for sp in page.get_texttrace():
        for u, gid, origin, _bbox in sp["chars"]:
            gid_at[(round(origin[0], 1), round(origin[1], 1))] = (sp["font"], gid)
    memo = cache.setdefault("__memo__", {})
    for ch, font in bad:
        hit = gid_at.get((round(ch["origin"][0], 1), round(ch["origin"][1], 1)))
        if not hit:
            continue
        if hit in memo:
            c = memo[hit]
        else:
            c = None
            fg = _font_glyphs(page.parent, hit[0], cache)
            if fg and 0 <= hit[1] < len(fg[0]):
                name = fg[0][hit[1]]
                greek = bool(GREEK_FONT_RE.search(hit[0]))
                if not greek and len(name) > 1:  # "a", "S": a reused Latin name says nothing
                    c = _name_to_char(name)
                    if c == ch["c"] and font in symbolish:
                        c = None
                if c is None:
                    c = match_outline(fg[0], fg[1], hit[1], greek_only=greek and len(name) == 1)
                    if not greek and c in ("Σ", "Π"):
                        c = {"Σ": "∑", "Π": "∏"}[c]  # in a symbol font these are the big operators
            memo[hit] = c
        if c:
            ch["c"] = c


def split_equation_number(glyphs: list[Glyph]) -> tuple[list[Glyph], str]:
    """Remove a right-margin equation number like "(2.18)" from a formula.
    It is the last group of glyphs, set apart by a wide gap."""
    vis = sorted((g for g in glyphs if not g.c.isspace()), key=lambda g: g.x0)
    for k in range(len(vis) - 1, 0, -1):
        if vis[k].x0 - vis[k - 1].x1 > 3 * vis[k].size:
            tail = "".join(g.c for g in vis[k:])
            if EQ_NUM_RE.match(tail):
                return vis[:k], tail[1:-1]
            break
    return glyphs, ""
