"""Page text extraction with a garbled-text check and an OCR fallback.

Why text extraction fails on "weird font" PDFs: a PDF stores glyph codes, not
letters. Readers map codes back to Unicode through the font's ToUnicode CMap
(ISO 32000-1, section 9.10). Subset or custom-encoded fonts often ship without
a correct CMap, so extraction returns symbols, U+FFFD, or shifted letters.
The glyphs still *look* right, so rendering the page and running OCR recovers
the words.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import pymupdf

from . import math as mth
from . import mathlayout as ml
from .wordlist import COMMON_WORDS

# Tesseract's docs say accuracy drops below about 300 DPI
# (https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html).
OCR_DPI = 300

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class Line:
    text: str
    size: float  # font size in points (text layer) or glyph height in points (OCR)
    y0: float  # top of the line, as a fraction of page height
    bold: bool = False
    bbox: tuple = (0, 0, 0, 0)  # in PDF points
    conf: float = 100.0  # OCR confidence 0-100 (100 for text layer)
    math: bool = False
    glyphs: list = field(default_factory=list, repr=False)  # text layer only
    junk: bool = False  # has math glyphs that decode to wrong characters
    block: int = -1  # PyMuPDF text block on the page (-1: unknown, e.g. OCR); keeps a caption apart from a side box
    caption: str = ""  # "Figure 1.11" / "Table 2.1" / "Example 2.1" when this line is such a label
    role: str = "body"  # "caption" (a figure/table caption block), "aside" (small text beside it) or "body"


@dataclass
class Page:
    number: int  # 0-based
    lines: list[Line] = field(default_factory=list)
    source: str = "text"  # "text" or "ocr"
    quality: float = 0.0
    rotated: int = 0  # degrees the scan was turned before OCR

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


def text_quality(text: str) -> float:
    """Score 0..1 for how much `text` looks like real English prose.

    Two signals:
    * Bad characters: U+FFFD, private-use and control characters appear when a
      font has no usable ToUnicode map.
    * Common-word coverage: in normal English the 100 most frequent words make
      up roughly half of all running words (Zipf's law; the Oxford English
      Corpus reports ~50%). Garbled extraction produces "words" that are almost
      never in that list.
    """
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    bad = sum(
        1
        for c in stripped
        if c == "�" or 0xE000 <= ord(c) <= 0xF8FF or (ord(c) < 32)
    )
    bad_ratio = bad / len(stripped)
    words = [w.lower().strip("'") for w in _WORD_RE.findall(text)]
    if len(words) < 8:
        # Too little text for word statistics (title pages, captions): accept
        # it if it contains any common word, otherwise treat it as suspect so
        # OCR gets a chance to do better.
        base = 0.6 if any(w in COMMON_WORDS for w in words) else 0.3
        return max(0.0, 1.0 - 5 * bad_ratio) * base
    coverage = sum(1 for w in words if w in COMMON_WORDS) / len(words)
    letters = sum(len(w) for w in words) / len(stripped)
    # Real prose: coverage ~0.35-0.55 for top-200 list, letters ~0.8+.
    score = min(1.0, coverage / 0.30) * 0.7 + min(1.0, letters / 0.75) * 0.3
    return max(0.0, score - 3 * bad_ratio)


_FONT_NAMES: dict = {}  # per-document glyph-name cache


# "F I G U R E 1. 11", "■F I G U R E P 2 . 1 5", "T A B L E 1. 1", "E X A M P L E 2 . 1": letter-spaced labels
# on a line of their own, as TeX-style textbooks set them.
_LABEL_RE = re.compile(
    r"^[^A-Za-z0-9]*(F\s?I\s?G\s?U\s?R\s?E|T\s?A\s?B\s?L\s?E|E\s?X\s?A\s?M\s?P\s?L\s?E)\s*"
    r"((?:P\s?)?(?:\d\s?){1,2})\s?\.\s?((?:\d\s?){1,3}[a-z]?)[^A-Za-z0-9]*$",
    re.IGNORECASE,
)
# "Figure 1.2 Cutaway of the engine" at the start of a block, for books that do not letter-space labels.
_INLINE_LABEL_RE = re.compile(r"^(Figure|Fig\.|Table)\s+(P?\d+[.\-]\d+)[a-z]?[.:]?\s+\S", re.IGNORECASE)


def _label(plain: str) -> tuple[str, str] | None:
    m = _LABEL_RE.match(plain)
    if not m:
        return None
    ident = re.sub(r"\s", "", m.group(2)) + "." + re.sub(r"\s", "", m.group(3))
    return re.sub(r"\s", "", m.group(1)).capitalize(), ident


def _lines_from_text_layer(page: pymupdf.Page, math: bool = True) -> list[Line]:
    h = page.rect.height or 1
    out: list[Line] = []
    d = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE, sort=True)
    if math:
        ml.repair_unknown_glyphs(page, d, _FONT_NAMES.setdefault(id(page.parent), {}))
    raw_lines = []
    for bi, block in enumerate(d["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            glyphs = ml.glyphs_of_line(line)
            if any(not g.c.isspace() for g in glyphs):
                raw_lines.append([list(line["bbox"]), glyphs, bi])
    if math:
        raw_lines = _merge_row_fragments(raw_lines)
    seen_blocks: set[int] = set()
    sizes = sorted(max(g.size for g in gs if not g.c.isspace()) for _b, gs, _i in raw_lines)
    page_size = sizes[len(sizes) // 2] if sizes else 10.0
    for bbox, glyphs, blk in raw_lines:
        first_in_block = blk not in seen_blocks
        seen_blocks.add(blk)
        vis = [g for g in glyphs if not g.c.isspace()]
        plain = "".join(g.c for g in glyphs).strip()
        # Weight size/bold by character count. Superscripts and footnote
        # marks are excluded from the size so they don't shrink headings.
        big = max(g.size for g in vis)
        main = [g for g in vis if g.size >= 0.85 * big]
        size = sum(g.size for g in main) / len(main)
        bold = sum(1 for g in main if re.search(r"Bold|Black|Heavy|Semibold", g.font)) > len(main) / 2
        if math:
            text, is_math = ml.line_text(glyphs)
            junk = any(ml.is_junk(g) for g in vis)
            if not text:
                continue
        else:
            text, is_math, junk = plain, False, False
        caption = ""
        lab = _label(plain)
        if lab:  # say "Figure 1.11." instead of spelling out F-I-G-U-R-E, and keep it out of the math pass
            text, is_math, junk, caption = f"{lab[0]} {lab[1]}.", False, False, f"{lab[0]} {lab[1]}"
        elif first_in_block and (m := _INLINE_LABEL_RE.match(plain)) and (bold or size < 0.95 * page_size):
            caption = f"{'Table' if m.group(1).lower() == 'table' else 'Figure'} {m.group(2).upper()}"
        out.append(Line(text, round(size, 1), bbox[1] / h, bold, tuple(bbox),
                        100.0, is_math, glyphs if math else [], junk, blk, caption))
    return out


def _glyphs_overlap(a: list, b: list) -> bool:
    va = [g for g in a if not g.c.isspace()]
    vb = [g for g in b if not g.c.isspace()]
    return any(g.x0 < h.x1 - 0.5 and h.x0 < g.x1 - 0.5 for g in va for h in vb)


def _merge_row_fragments(raw_lines: list) -> list:
    """PDF producers often split one printed row into several text lines,
    especially around math (a tall bracket, a script, a symbol from another
    font). Merge pieces that share a row and don't overlap horizontally."""
    def prose(gs: list) -> bool:
        return len(re.findall(r"[A-Za-z]{3,}", "".join(g.c for g in gs))) >= 2

    merged: list = []
    for bbox, glyphs, blk in sorted(raw_lines, key=lambda x: (x[0][1], x[0][0])):
        for m in merged:
            mb = m[0]
            hmin = min(bbox[3] - bbox[1], mb[3] - mb[1])
            vover = min(bbox[3], mb[3]) - max(bbox[1], mb[1])
            hgap = max(bbox[0] - mb[2], mb[0] - bbox[2])
            eqnum = ml.EQ_NUM_RE.match("".join(g.c for g in glyphs).strip())
            # Prose from two different blocks (a caption and the box beside it) is not one row.
            other_prose = blk not in {f[2] for f in m[2:]} and prose(glyphs) and prose(m[1])
            if (hmin > 0 and vover > 0.5 * hmin and (hgap < 40 or eqnum) and not other_prose
                    and not _glyphs_overlap(glyphs, m[1])):
                m.append((bbox[0], glyphs, blk))
                m[1] = m[1] + glyphs
                m[0] = [min(mb[0], bbox[0]), min(mb[1], bbox[1]), max(mb[2], bbox[2]), max(mb[3], bbox[3])]
                break
        else:
            merged.append([list(bbox), list(glyphs), (bbox[0], glyphs, blk)])
    # Rebuild each row from its fragments left to right, keeping the order
    # of glyphs inside a fragment (ligatures share one box). The row belongs
    # to the block that supplied most of its glyphs.
    out = []
    for m in merged:
        frags = sorted(m[2:], key=lambda f: f[0])
        blk = max(frags, key=lambda f: len(f[1]))[2]
        out.append([m[0], [g for _, gs, _b in frags for g in gs], blk])
    return out


def _bars(page: pymupdf.Page) -> list[tuple]:
    """Thin horizontal rules (fraction bars, radical overlines) on the page.

    They may be vector paths or, from some TeX drivers, 1-pixel image masks,
    so take every painted item from the page's bbox log.
    """
    out = []
    for kind, r in page.get_bboxlog():
        if "text" in kind:
            continue
        r = pymupdf.Rect(r)
        if r.height <= 1.5 and r.width >= 3:
            out.append((r.x0, (r.y0 + r.y1) / 2, r.x1))
    return out


def _figure_ink(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Bounding boxes of drawn strokes that typeset math never uses.

    In a typeset formula the only vector paths are horizontal rules
    (fraction bars, radical overlines); brackets, radical signs and
    operators are font glyphs. Curves, vertical strokes and slanted strokes
    belong to diagrams and plots. Measured on 20 regions of Craig's
    *Mechanics of Materials* that the detector flagged as math: the 7 real
    formulas had 0 such strokes each, and 12 of the 13 figures had 12 or
    more. This is the text/graphics separation step of page segmentation
    (Fletcher & Kasturi, IEEE PAMI 10(6), 1988), done on vector paths
    instead of pixels.
    """
    out = []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "c":
                out.append((pymupdf.Rect(it[1], it[1]) | it[2] | it[3] | it[4]) + (-0.5, -0.5, 0.5, 0.5))
            elif it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.y - b.y) >= 0.5:  # anything but a horizontal rule
                    out.append((pymupdf.Rect(a, a) | b) + (-0.5, -0.5, 0.5, 0.5))
    return out


def _drop_figure_labels(lines: list[Line], idx: list[int], ink: list[pymupdf.Rect]) -> list[int]:
    """Indices of `idx` that are formula lines rather than figure labels.

    A label ("A", "L*", "(b)", "30°C") sits within a few points of the
    strokes it names; a displayed formula is set apart from any diagram.
    A line is a label when 2+ figure strokes come within 6 pt of it.
    """
    keep = []
    for k in idx:
        lr = pymupdf.Rect(lines[k].bbox) + (-6, -6, 6, 6)
        if sum(1 for r in ink if lr.intersects(r)) < 2:
            keep.append(k)
    return keep


_RELATION_RE = re.compile(r"[=<>≤≥≈≠≡∝→⇒]|\\(int|sum|prod|frac|sqrt|lim)")


def _is_bare_label(tex: str) -> bool:
    """A displayed "formula" that is only a few symbols and has no relation
    or operator ("A", "(a)(b)", "40", "σ_{PL}") is a figure label or axis
    tick, not something to read aloud."""
    if _RELATION_RE.search(tex):
        return False
    vis = re.sub(r"\\[a-zA-Z]+|[{}^_ ]", "", tex)
    return len(vis) <= 8


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def detect_rotation(page: pymupdf.Page) -> int:
    """Degrees to rotate a scanned page clockwise so text is upright, using
    Tesseract's orientation and script detection (--psm 0). Phone and
    flatbed scans of books are often sideways, and OCR on a sideways page
    returns gibberish. Returns 0 when unsure (confidence below 2)."""
    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    with tempfile.TemporaryDirectory() as tmp:
        img = os.path.join(tmp, "page.png")
        pix.save(img)
        res = subprocess.run(["tesseract", img, "stdout", "--psm", "0", "--dpi", "150"],
                             capture_output=True, text=True)
    rot = re.search(r"Rotate:\s*(\d+)", res.stdout)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", res.stdout)
    if rot and conf and float(conf.group(1)) >= 2:
        return int(rot.group(1)) % 360
    return 0


def _lines_from_ocr(page: pymupdf.Page, lang: str = "eng", rotate: int = 0) -> list[Line]:
    """Render the page at OCR_DPI and run Tesseract, keeping line geometry.
    With `rotate`, the image is turned first and geometry is in the turned
    frame (fine for reading order and headings)."""
    mat = pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72).prerotate(rotate)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
    with tempfile.TemporaryDirectory() as tmp:
        img = os.path.join(tmp, "page.png")
        pix.save(img)
        # --psm 3: fully automatic page segmentation (Tesseract default).
        res = subprocess.run(
            ["tesseract", img, "stdout", "-l", lang, "--psm", "3", "tsv"],
            capture_output=True,
            text=True,
            check=True,
        )
    px_to_pt = 72 / OCR_DPI
    rows = csv.DictReader(io.StringIO(res.stdout), delimiter="\t", quoting=csv.QUOTE_NONE)
    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("level") != "5" or not (r.get("text") or "").strip():
            continue
        if float(r.get("conf", -1)) < 0:
            continue
        key = (int(r["block_num"]), int(r["par_num"]), int(r["line_num"]))
        grouped.setdefault(key, []).append(r)
    out = []
    for key in sorted(grouped):
        words = grouped[key]
        text = " ".join(w["text"] for w in words)
        top = min(int(w["top"]) for w in words)
        # Median word height approximates the font's cap+descender height,
        # which scales with font size, so it can drive heading detection.
        heights = sorted(int(w["height"]) for w in words)
        size = heights[len(heights) // 2] * px_to_pt
        conf = sum(float(w["conf"]) for w in words) / len(words)
        bbox = (
            min(int(w["left"]) for w in words) * px_to_pt,
            top * px_to_pt,
            max(int(w["left"]) + int(w["width"]) for w in words) * px_to_pt,
            max(int(w["top"]) + int(w["height"]) for w in words) * px_to_pt,
        )
        out.append(Line(text, round(size, 1), top / pix.height, False, bbox, conf, mth.ocr_line_is_math(text, conf)))
    return out


def _prose_like(l: Line) -> bool:
    t = re.sub(r"\$[^$]*\$", " ", l.text)  # ignore inline math
    return any(len(w) >= 4 for w in _WORD_RE.findall(t)) or len(t.strip()) > 25


def _mark_ocr_math(page: pymupdf.Page, lines: list[Line]) -> list[Line]:
    """On OCR pages Tesseract skips or mangles equations, so its own lines
    can't be trusted to locate them. Instead, treat ink that confident prose
    lines don't explain as math, and drop the OCR lines that overlap it."""
    prose = [l.bbox for l in lines if not l.math and l.conf >= 60]
    regions = [r for r in mth.unexplained_ink(page, prose) if (r[3] - r[1]) < 0.3 * page.rect.height]
    keep = [l for l in lines if not any(mth._overlap(l.bbox, r) > 0.3 for r in regions) and not l.math]
    for r in regions:
        keep.append(Line("", 0, r[1] / page.rect.height, False, r, 0, True))
    keep.sort(key=lambda l: (round(l.bbox[1]), l.bbox[0]))
    return keep


_TRAILING_EQ_NUM_RE = re.compile(r"\(\d+(\.\d+)*(\.?[a-z])?\)\s*$")


def _promote_display_lines(lines: list[Line]) -> None:
    """Mark set-off formulas written in the body font as math.

    Books set in one text font (Palatino, Times) print equations in that same
    font, so only the inline math is found ("I $=\\lim ∆Q=$ dQ"). A display
    equation is indented from both margins, or carries an equation number at
    the right, and contains a relation or inline math; a prose line runs
    margin to margin.
    """
    body = [l for l in lines if not l.math and len(l.text) > 40]
    if len(body) < 3:
        return
    left = sorted(l.bbox[0] for l in body)[len(body) // 2]
    right = sorted(l.bbox[2] for l in body)[len(body) // 2]
    for l in lines:
        if l.math or l.caption or not ("$" in l.text or "=" in l.text):
            continue
        em = max(l.size, 6)
        indented = l.bbox[0] > left + 3 * em and l.bbox[2] < right - 3 * em
        numbered = bool(_TRAILING_EQ_NUM_RE.search(l.text)) and "=" in l.text
        if indented or numbered:
            l.math = True


def recognize_math(page: pymupdf.Page, lines: list[Line], ocr: bool = False, log=None) -> list[Line]:
    """Replace each group of equation lines with one `$$LaTeX$$` line."""
    if ocr:
        lines = _mark_ocr_math(page, lines)
    else:
        _promote_display_lines(lines)
    # Fraction pieces like "2a" or a lone radical sign are too short to
    # classify alone; pull in short non-prose lines that touch a math line.
    for _ in range(2):
        for i, l in enumerate(lines):
            if l.math or l.caption or _prose_like(l) or ml.EQ_NUM_RE.match(l.text.strip()):
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(lines) and lines[j].math:
                    nb = lines[j]
                    h = max(nb.bbox[3] - nb.bbox[1], 1)
                    if abs(l.bbox[1] - nb.bbox[1]) < 2.5 * h:
                        l.math = True
    groups = mth.group_regions(lines, lambda l: l.math)
    if not groups:
        return lines
    have_ocr = mth.MathOCR.available() and not mth.MathOCR.disabled
    ink = [] if ocr else _figure_ink(page)
    out, done = [], set()
    starts = {g[0]: g for g in groups}
    for i, l in enumerate(lines):
        if i in done:
            continue
        g = starts.get(i)
        if g is None:
            out.append(l)
            continue
        done.update(g)
        # Labels inside a diagram are not read aloud; a formula beside the
        # diagram is kept.
        g = _drop_figure_labels(lines, g, ink) if ink else g
        if not g:
            continue
        l = lines[g[0]]
        rect = pymupdf.Rect(lines[g[0]].bbox)
        for k in g[1:]:
            rect |= pymupdf.Rect(lines[k].bbox)
        sizes = sorted(x.size for x in lines if not x.math) or [l.size]
        size = sizes[len(sizes) // 2]  # body size, so equations never look like headings
        glyphs = [gl for k in g for gl in lines[k].glyphs]
        junk = any(lines[k].junk for k in g)
        bars = [b for b in _bars(page) if rect.y0 - 2 <= b[1] <= rect.y1 + 2
                and b[0] >= rect.x0 - 15 and b[2] <= rect.x1 + 15] if glyphs else []
        if glyphs and ml.is_table(glyphs, size, rect.width):
            out.append(Line("[table]", size, l.y0, False, tuple(rect), 100.0, True, block=l.block))
            continue
        if glyphs and not junk and not ml.is_complex(glyphs, bars, size):
            # Text layer with trustworthy Unicode: rebuild from geometry,
            # one band (one printed formula row, fractions included) at a time.
            for bg, bb in ml.bands(glyphs, bars):
                bg, num = ml.split_equation_number(bg)
                tex = ml.region_latex(bg, bb)
                tail = f" (Equation {num})" if num else ""
                tex = tex.rstrip(" .,;")
                if tex and (num or not _is_bare_label(tex)):
                    y0 = min(x.y0 for x in bg) / page.rect.height
                    out.append(Line(f"$${tex}$$" + tail, size, y0, False, tuple(rect), 100.0, True, block=l.block))
            continue
        # Formulas contain no curved, vertical or slanted strokes; 4+ inside
        # the region means a diagram or plot (see _figure_ink).
        if sum(1 for r in ink if r in rect + (-2, -2, 2, 2)) >= 4:
            continue
        tex, tail, crop = "", "", rect
        if glyphs:
            # The image model returns nothing for a crop that includes the
            # equation number far out at the right margin; crop it off and
            # say it after the formula.
            rest, num = ml.split_equation_number(glyphs)
            if num and rest:
                crop = pymupdf.Rect(min(g.x0 for g in rest), rect.y0, max(g.x1 for g in rest), rect.y1)
                tail = f" (Equation {num})"
        # Very tall regions are figures or tables caught by the detector, not
        # formulas; the image model is slow on them and rarely right.
        if have_ocr and rect.height < 0.2 * page.rect.height:
            try:
                tex = mth.MathOCR.recognize(page, crop)
                if glyphs:  # even a "junk" formula's "=" comes from the plain roman font
                    tex = mth.reconcile_relations(tex, "".join(gl.c for gl in glyphs))
            except Exception as e:  # keep going on a bad crop
                if log:
                    log(f"math OCR failed on page {page.number + 1}: {e}")
        text = f"$${tex}$$" + tail if mth.looks_like_latex(tex) else "[equation]"
        out.append(Line(text, size, l.y0, False, tuple(rect), 100.0, True, block=l.block))
    return out


# Reflowable ebooks (EPUB, MOBI, FB2) have no pages of their own; MuPDF lays
# them out at a page size we choose. 270 x 585 pt matches a phone screen's
# 1 : 2.17 shape (iPhone 15: 1179 x 2556 px), and 270 pt shown across the
# phone's 390 CSS px makes 11 pt body text about 16 px, Apple Books' own
# default text size range.
EBOOK_PAGE = (270, 585, 11)


def open_doc(path: str) -> pymupdf.Document:
    doc = pymupdf.open(path)
    if doc.is_reflowable:
        w, h, size = EBOOK_PAGE
        doc.layout(width=w, height=h, fontsize=size)
    return doc


def extract_pages(
    path: str,
    mode: str = "auto",
    threshold: float = 0.45,
    lang: str = "eng",
    progress=None,
    math: bool = True,
) -> list[Page]:
    """Extract every page.

    mode="auto": use the text layer unless its quality score is below
    `threshold`, then OCR and keep whichever result scores higher.
    mode="text": never OCR.  mode="ocr": always OCR.
    math=True: find equations and replace them with $$LaTeX$$ lines.
    """
    doc = open_doc(path)
    can_ocr = tesseract_available()
    if mode == "ocr" and not can_ocr:
        raise RuntimeError("Tesseract is not installed; see README for install steps.")
    pages = []
    for i, pg in enumerate(doc):
        lines = _lines_from_text_layer(pg, math) if mode != "ocr" else []
        q = text_quality("\n".join(l.text for l in lines))
        page = Page(i, lines, "text", q)
        if can_ocr and (mode == "ocr" or (mode == "auto" and q < threshold)):
            o_lines = _lines_from_ocr(pg, lang)
            oq = text_quality("\n".join(l.text for l in o_lines))
            rotated = 0
            if oq < threshold:
                rot = detect_rotation(pg)
                if rot:
                    r_lines = _lines_from_ocr(pg, lang, rot)
                    rq = text_quality("\n".join(l.text for l in r_lines))
                    if rq > oq:
                        o_lines, oq, rotated = r_lines, rq, rot
            if mode == "ocr" or oq > q:
                page = Page(i, o_lines, "ocr", oq)
                page.rotated = rotated
        if math and not getattr(page, "rotated", 0) and (page.source == "text" or page.quality >= 0.6):
            # Equation search on OCR pages needs readable prose to tell
            # equations apart from text Tesseract could not read.
            page.lines = recognize_math(pg, page.lines, ocr=page.source == "ocr")
        if page.source == "text":  # block by block (the math pass needs lines in row order, prose does not)
            page.lines.sort(key=lambda l: (l.block, l.bbox[1], l.bbox[0]))
        pages.append(page)
        if progress:
            progress(i + 1, len(doc), page)
    return pages
