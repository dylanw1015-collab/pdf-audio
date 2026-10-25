"""End-to-end checks on the three sample books (see make_samples.py).

Run: python -m pytest tests -q
"""

import shutil

import pytest

from pdfaudiobook.book import load_book
from pdfaudiobook.extract import text_quality
from pdfaudiobook.tts import chunk_text

from make_samples import CHAPTERS, PARA, break_tounicode, rasterize, write_book

EXPECTED_TITLES = [f"{h}: {s}" for h, s in CHAPTERS]


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    d = tmp_path_factory.mktemp("samples")
    write_book(str(d / "clean.pdf"))
    pdfmetrics.registerFont(TTFont("DVS", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
    pdfmetrics.registerFont(TTFont("DVSB", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"))
    write_book(str(d / "garbled_truth.pdf"), "DVS", "DVSB", outline=False)
    break_tounicode(str(d / "garbled_truth.pdf"), str(d / "garbled.pdf"))
    rasterize(str(d / "clean.pdf"), str(d / "scanned.pdf"))
    return d


def word_accuracy(got: str, want: str) -> float:
    """1 - word error rate (WER = word-level Levenshtein distance / reference
    length, the standard ASR/OCR metric)."""
    g, w = got.lower().split(), want.lower().split()
    prev = list(range(len(w) + 1))
    for i, gw in enumerate(g, 1):
        cur = [i]
        for j, ww in enumerate(w, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (gw != ww)))
        prev = cur
    return 1 - prev[-1] / len(w)


def test_quality_score_separates_prose_from_garble():
    assert text_quality(PARA * 2) > 0.9
    shifted = "".join(chr((ord(c) - 97 + 7) % 26 + 97) if c.islower() else c for c in PARA * 2)
    assert text_quality(shifted) < 0.45  # below the OCR threshold
    assert text_quality("�" * 40) == 0


def test_chunks_stay_short():
    chunks = chunk_text(PARA * 10, max_chars=400)
    assert all(len(c) <= 400 for c in chunks)
    assert " ".join(chunks).split() == (PARA * 10).split()


needs_tesseract = pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")


@pytest.mark.parametrize(
    "name,strategy,ocr",
    [
        ("clean", "outline", False),
        pytest.param("garbled", "headings", True, marks=needs_tesseract),
        pytest.param("scanned", "headings", True, marks=needs_tesseract),
    ],
)
def test_book(samples, name, strategy, ocr):
    b = load_book(str(samples / f"{name}.pdf"), use_cache=False)
    assert b["strategy"] == strategy
    assert bool(b["ocr_pages"]) == ocr
    assert [c.title for c in b["chapters"]] == EXPECTED_TITLES
    for c, (head, sub) in zip(b["chapters"], CHAPTERS):
        # Heading lines, then body. Running header and page numbers removed.
        assert c.paragraphs[0] == head and c.paragraphs[1] == sub
        body = " ".join(c.paragraphs[2:])
        assert "QUIET TOWN" not in body
        assert word_accuracy(body, reference_body(samples, name, c)) > 0.97


def reference_body(samples, name, ch):
    """Ground truth: the un-broken PDF's own text layer for the same pages."""
    import pymupdf

    doc = pymupdf.open(str(samples / ("garbled_truth.pdf" if name == "garbled" else "clean.pdf")))
    lines = []
    for p in range(ch.start_page, ch.end_page + 1):
        lines += [l for l in doc[p].get_text().splitlines() if l.strip() and l != "THE QUIET TOWN"
                  and not l.strip().isdigit()]
    words = " ".join(lines[2:]).split()  # drop the two heading lines
    out = " ".join(words)
    return out.replace("- ", "")


# ---------------------------------------------------------------- equations

from pdfaudiobook.math import MathOCR, MathSpeaker, latex_to_words  # noqa: E402


def test_builtin_math_words():
    assert latex_to_words(r"E = mc^2") == "E equals m c squared"
    w = latex_to_words(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
    assert "the fraction" in w and "plus or minus" in w and "square root of b squared" in w


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_sre_math_speech():
    s = MathSpeaker()
    assert s.engine == "sre"
    assert s.speak(r"\frac{1}{2}") == "one half"
    assert s.speak_text("so $$E=mc^2$$ holds") == "so E equals m c squared. holds"


@pytest.fixture(scope="session")
def math_samples(samples):
    from make_samples import write_math_book

    write_math_book(str(samples / "math.pdf"))
    rasterize(str(samples / "math.pdf"), str(samples / "math_scanned.pdf"), dpi=300)
    return samples


@needs_tesseract
@pytest.mark.parametrize("name", ["math", "math_scanned"])
def test_equations_found_and_chapters_kept(math_samples, name):
    b = load_book(str(math_samples / f"{name}.pdf"), use_cache=False)
    assert [c.title for c in b["chapters"]] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    eqs = [p for c in b["chapters"] for p in c.paragraphs if p.startswith("$$") or p == "[equation]"]
    assert len(eqs) == 3
    if MathOCR.available():
        assert eqs[1] in ("$$E=mc^{2}$$", "$$E=m c^{2}$$")


@needs_tesseract
def test_no_false_equations_in_prose_scan(samples):
    b = load_book(str(samples / "scanned.pdf"), use_cache=False)
    assert not any("$$" in c.text or "[equation]" in c.text for c in b["chapters"])


# ---------------------------------------------------------------- real-book fixes

def test_sideways_scan_is_rotated(samples):
    """Scans are often sideways; orientation detection must fix them."""
    import pymupdf

    src = pymupdf.open(str(samples / "scanned.pdf"))
    out = pymupdf.open()
    out.insert_pdf(src, from_page=1, to_page=1)
    out[0].set_rotation(180)  # upside down: Tesseract's layout step alone can't fix this
    rot = samples / "sideways.pdf"
    out.save(str(rot))
    # Bake the rotation into the image so the page itself is sideways.
    baked = pymupdf.open()
    pix = pymupdf.open(str(rot))[0].get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    pg = baked.new_page(width=pix.width * 72 / 200, height=pix.height * 72 / 200)
    pg.insert_image(pg.rect, pixmap=pix)
    baked.save(str(samples / "sideways_baked.pdf"))
    from pdfaudiobook.extract import extract_pages

    (page,) = extract_pages(str(samples / "sideways_baked.pdf"), math=False)
    assert page.rotated == 180
    assert "bright cold day" in page.text


def test_skip_front_matter():
    from pdfaudiobook.chapters import Chapter, is_skippable

    for t in ["Cover", "Copyright Page", "CONTENTS", "Table of Contents", "Index", "Inside Front Cover", "9781441979186"]:
        assert is_skippable(Chapter(t, 0, 0))
    for t in ["Preface", "Chapter 1 Introduction", "Indexing Methods", "Contents of the Soul"]:
        assert not is_skippable(Chapter(t, 0, 0))


def test_auto_bookmark_depth():
    from pdfaudiobook.chapters import _auto_level

    # Sociobiology-like outline: book title, then Parts, then chapters.
    toc = [[1, "Book", 1]] + [[2, f"Part {p}", 1 + 100 * p] for p in range(3)]
    toc += [[3, f"Ch {c}", 1 + 12 * c] for c in range(25)]
    assert _auto_level(toc, 300) == 3
    # Textbook-like: chapters at the top level are already chapter-sized.
    toc = [[1, f"Ch {c}", 1 + 25 * c] for c in range(30)] + [[2, "s", 5]] * 100
    assert _auto_level(toc, 750) == 1


def test_geometric_math_rebuild():
    """Superscripts, subscripts, accents and fractions from glyph positions."""
    from pdfaudiobook.mathlayout import Glyph, line_text, region_latex

    def g(c, x, size=10.0, dy=0.0, font="Times-Italic"):
        return Glyph(c, x, 100 + dy - size, x + 0.5 * size, 100 + dy + 0.2 * size, 100 + dy, size, font)

    # x(t) = c e^{-at}  (exponent smaller and raised)
    row = [g("x", 0), g("(", 5, font="CMR10"), g("t", 9), g(")", 13, font="CMR10"), g(" ", 17),
           g("=", 20, font="CMR10"), g(" ", 26), g("c", 30), g("e", 35, font="Times-Roman"),
           g("−", 40, 7, -4, "CMSY10"), g("α", 44, 7, -4, "Symbol"), g("t", 48, 7, -4)]
    text, display = line_text(row)
    assert display and text == "$$x(t)=ce^{−αt}$$"
    # x_0 and an accent above a letter
    assert line_text([g("x", 0), g("0", 5, 7, 2, "Times-Roman")])[0] == "$$x_{0}$$"
    assert line_text([g("x", 0), g("˙", 1, font="CMR10")])[0] == "$$\\dot{x}$$"
    # 2/13 with a fraction bar at y=96
    num = [Glyph("2", 10, 84, 15, 94, 93, 10, "Times-Roman")]
    den = [Glyph("1", 8, 98, 13, 108, 107, 10, "Times-Roman"), Glyph("3", 13, 98, 18, 108, 107, 10, "Times-Roman")]
    assert region_latex(num + den, [(7, 96, 19)]) == "\\frac{2}{13}"


def test_delta_is_a_shape_reference():
    """The AGL maps the glyph name "Delta" to U+2206 (increment), so Δ was
    missing from the shape references and Craig's Δ came out as Λ."""
    import pymupdf

    from pdfaudiobook.glyphmatch import glyphset, scores_for

    order, gs = glyphset(pymupdf.Font("symb").buffer, "cff")
    assert scores_for(gs["Delta"])[0][1] == "Δ"
    assert scores_for(gs["Omega"])[0][1] == "Ω"


def test_figure_labels_are_not_formulas():
    import pymupdf

    from pdfaudiobook.extract import Line, _drop_figure_labels, _is_bare_label

    # A label sitting on a drawn curve is dropped; a formula 30 pt away is kept.
    lines = [Line("A", 10, 0.1, False, (100, 100, 106, 110)),
             Line("σ = F/A", 10, 0.2, False, (100, 150, 160, 160))]
    ink = [pymupdf.Rect(104, 105, 140, 120), pymupdf.Rect(90, 95, 103, 104)]
    assert _drop_figure_labels(lines, [0, 1], ink) == [1]
    assert _is_bare_label("(a)(b)") and _is_bare_label("σ_{PL}") and _is_bare_label("40")
    assert not _is_bare_label("σ=E ε") and not _is_bare_label("\\int_{A} σ dA")


def test_image_model_nonsense_is_rejected():
    from pdfaudiobook.math import looks_like_latex

    # pix2tex output for a crane diagram in Craig, ch. 2
    junk = "\\stackrel{\\cal D}{\\P}_{\\S}^{3}{}_{\\star_{\\star}^{+}}_{\\star}^{\\phantom{0}}"
    assert not looks_like_latex(junk)
    assert looks_like_latex("\\sigma_{1}=\\frac{F_{1}}{A_{1}}=-63.7\\,\\mathrm{MPa}")


def test_subscript_product_is_not_a_function():
    from pdfaudiobook.math import _group_subscripts

    assert _group_subscripts("C_{1}e^{2t}") == "{C_{1}}e^{2t}"
    assert _group_subscripts("x(t)=x_{0}e^{t}") == "x(t)=x_{0}\\cdot e^{t}"
    assert _group_subscripts("x_{p}(t)") == "x_{p}(t)"  # a real function of t
    assert _group_subscripts("\\sum_{i}a_{i}") == "\\sum_{i}a_{i}"


def test_chapter_title_not_read_twice():
    from pdfaudiobook.chapters import Chapter, spoken_text

    assert spoken_text(Chapter("Chapter 2", 0, 0, ["Chapter 2", "Text."])) == "Chapter 2\n\nText."
    assert spoken_text(Chapter("Stress", 0, 0, ["Text."])).startswith("Stress.")
