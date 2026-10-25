"""Identify symbols whose font gives no usable name, by their shape.

Some publisher math fonts (Linotype Mathematical Pi, custom "Grk" fonts)
name glyphs with opaque codes like "H11001", or with Latin letters, so
neither the Unicode map nor the glyph name says which symbol is drawn.
The outline still does. Each unknown glyph's outline is read from the
embedded font, rasterized, and compared with the outlines of candidate
symbols in the Symbol font built into MuPDF, using normalized
cross-correlation on a fixed-size bitmap. This is classical
template-matching OCR (Duda & Hart, *Pattern Classification and Scene
Analysis*, 1973, ch. 7). It works well here because the inputs are clean
vector outlines, not scans, and the candidate set is small. A glyph is
only renamed when one candidate clearly wins; otherwise it is left alone.

Results are cached per (font, glyph id), so each distinct glyph in a book
is matched once.
"""

from __future__ import annotations

import io

import numpy as np

CANDIDATES = (
    "+−×÷=≠≈≡∼<>≤≥±°′″∞∂∇∑∏∫√∈∉⊂⊃⊆∪∩→←↑↓↔⇒⇐⇔∝∠⊥·•…/|()[]{}"
    "αβγδεζηθικλμνξοπρστυφχψωϕϑΓΔΘΛΞΠΣΦΨΩ"
)
GREEK = set("αβγδεζηθικλμνξοπρστυφχψωϕϑΓΔΘΛΞΠΣΦΨΩ")
LATIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
N = 40  # bitmap side in pixels
_refs: list | None = None


class _PolyPen:
    """fontTools pen that flattens outlines into polygons."""

    def __init__(self):
        self.polys, self.cur = [], []

    def moveTo(self, p):
        self.cur = [p]

    def lineTo(self, p):
        self.cur.append(p)

    def curveTo(self, *pts):
        p0 = self.cur[-1]
        if len(pts) == 3:  # cubic Bezier
            c1, c2, p3 = pts
            for t in np.linspace(0.1, 1, 10):
                u = 1 - t
                self.cur.append((u**3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t**3 * p3[0],
                                 u**3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t**3 * p3[1]))
        else:
            self.cur.extend(pts)

    def qCurveTo(self, *pts):
        self.cur.extend(p for p in pts if p is not None)

    def closePath(self):
        if len(self.cur) > 2:
            self.polys.append(self.cur)
        self.cur = []

    endPath = closePath

    def addComponent(self, *a):
        pass


def _raster(glyph, shear: float = 0.0):
    """Rasterize a fontTools glyph; return (normalized N*N vector, aspect).
    `shear` slants the outline (x += shear * y) to imitate italics."""
    import cv2

    pen = _PolyPen()
    glyph.draw(pen)
    if not pen.polys:
        return None
    if shear:
        pen.polys = [[(x + shear * y, y) for x, y in p] for p in pen.polys]
    pts = np.concatenate([np.array(p, np.float64) for p in pen.polys])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    side = max(x1 - x0, y1 - y0) or 1
    scale = (N - 6) / side
    ox = (N - (x1 - x0) * scale) / 2
    oy = (N - (y1 - y0) * scale) / 2
    mask = np.zeros((N, N), np.uint8)
    for p in pen.polys:
        poly = np.array([((x - x0) * scale + ox, N - ((y - y0) * scale + oy)) for x, y in p], np.int32)
        tmp = np.zeros_like(mask)
        cv2.fillPoly(tmp, [poly], 1)
        mask ^= tmp  # even-odd rule keeps counters (holes) open
    v = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    aspect = (x1 - x0) / max(y1 - y0, 1)
    return (v / n, aspect) if n else None


def glyphset(buf: bytes, ext: str):
    """(glyph order, glyph set) of an embedded CFF or TrueType font."""
    if ext == "cff":
        from fontTools.cffLib import CFFFontSet

        cff = CFFFontSet()
        cff.decompile(io.BytesIO(buf), None)
        top = cff[cff.fontNames[0]]
        return top.getGlyphOrder(), top.CharStrings
    from fontTools.ttLib import TTFont

    f = TTFont(io.BytesIO(buf))
    return f.getGlyphOrder(), f.getGlyphSet()


def _references():
    global _refs
    if _refs is None:
        import pymupdf
        from fontTools import agl

        # The AGL maps Delta, Omega and mu to the math/unit code points
        # (increment, ohm, micro); we want the Greek letters.
        special = {"minus": "−", "theta1": "ϑ", "phi1": "ϕ", "sigma1": "ς", "omega1": "ϖ",
                   "Delta": "Δ", "Omega": "Ω", "mu": "μ"}
        _refs = []
        # Symbols from the built-in Symbol font, plus Latin letters and digits
        # from the built-in Times, so a real "a" is recognized as "a".
        sources = [(glyphset(pymupdf.Font("symb").buffer, "cff"), CANDIDATES),
                   (glyphset(pymupdf.Font("tiro").buffer, "cff"), LATIN)]
        for (order, gs), allowed in sources:
          for name in order:
            u = special.get(name) or agl.toUnicode(name)
            if u and u in allowed:
                # Upright and two italic slants, since book fonts set math
                # letters in italic and the reference font is upright.
                for sh in (0.0, 0.2, 0.33):
                    r = _raster(gs[name], sh)
                    if r:
                        _refs.append((u, *r))
        _refs = [r for r in _refs if r[0]]
    return _refs


def scores_for(glyph) -> list[tuple[float, str]]:
    r = _raster(glyph)
    if not r:
        return []
    v, a = r
    best: dict[str, float] = {}
    for u, rv, ra in _references():
        sc = float(v @ rv) - 0.15 * abs(np.log(max(a, 1e-3) / max(ra, 1e-3)))
        best[u] = max(best.get(u, -9), sc)
    return sorted(((sc, u) for u, sc in best.items()), reverse=True)


def match_outline(order, gs, gid: int, greek_only: bool = False,
                  min_score: float = 0.65, margin: float = 0.03) -> str | None:
    """Best-matching symbol for glyph `gid`, or None if no clear winner.

    Thresholds were set on Craig's *Mechanics of Materials* (Wiley), whose
    Greek font draws all 25 lowercase letters with Latin names: 19 of them
    match correctly at these settings and the rest are left unmatched.
    """
    if not 0 <= gid < len(order):
        return None
    try:
        sc = scores_for(gs[order[gid]])
    except Exception:
        return None
    if greek_only:
        sc = [x for x in sc if x[1] in GREEK]
    if len(sc) < 2:
        return None
    (best, u), (second, _) = sc[0], sc[1]
    return u if best >= min_score and best - second >= margin else None
