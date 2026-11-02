"""Read-along EPUB export: structure, sync timings and highlight geometry.

Run: python -m pytest tests -q
"""

import glob
import os
import re
import shutil
import subprocess
import wave
import zipfile

import pytest

from pdfaudiobook.book import load_book
from pdfaudiobook.epub import boxes_for, export_epub, sentences

from make_samples import write_book

needs_tools = pytest.mark.skipif(not (shutil.which("espeak-ng") and shutil.which("ffmpeg")),
                                 reason="needs espeak-ng and ffmpeg")
EPUBCHECK = os.environ.get("EPUBCHECK_JAR", "")


def test_sentences_keep_equations_whole():
    text = "First one. Then $$x = 1$$ and the end. Last."
    parts = [text[a:b] for a, b in sentences(text)]
    assert parts == ["First one.", "Then ", "$$x = 1$$", " and the end.", "Last."]


def test_boxes_interpolate_inside_a_line():
    geo = [[0, 100, 10, 300, 20, 0, 100], [1, 100, 10, 300, 20, 101, 150]]
    got = boxes_for(geo, 50, 120)
    assert got[0] == [(200.0, 10, 300.0, 20)]  # second half of line 1
    assert list(got) == [0, 1]  # carries over to the next page
    assert boxes_for([[-1, 0, 0, 1, 1, 0, 10]], 0, 5) == {}  # turned scan: no box


@needs_tools
@pytest.mark.parametrize("strips", [1, 3])
def test_epub_export(tmp_path, strips):
    pdf = str(tmp_path / "clean.pdf")
    write_book(pdf)
    book = load_book(pdf, use_cache=False, math=False)
    assert all(len(g) == len(c.paragraphs) for c in book["chapters"] for g in [c.geometry])

    class Espeak:
        def to_wav(self, text, path):
            subprocess.run(["espeak-ng", "-s", "400", "-w", path, text], check=True)

    out = export_epub(pdf, book, Espeak(), str(tmp_path / "out"), [1], strips=strips, log=lambda *a: None)
    with zipfile.ZipFile(out) as z:
        first = z.infolist()[0]
        assert first.filename == "mimetype" and first.compress_type == zipfile.ZIP_STORED
        z.extractall(tmp_path / "x")
    oebps = tmp_path / "x" / "OEBPS"
    opf = (oebps / "package.opf").read_text()
    assert "pre-paginated" in opf and "-epub-media-overlay-active" in opf

    # Every clip lies inside its audio file, in order, and every highlight
    # box lies inside its page.
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", str(oebps / "audio" / "ch001.m4a")]))
    clips = []
    for smil in sorted(glob.glob(str(oebps / "*.smil"))):
        for b, e in re.findall(r'clipBegin="([^"]+)" clipEnd="([^"]+)"', open(smil).read()):
            sec = lambda c: sum(float(x) * m for x, m in zip(c.split(":"), (3600, 60, 1)))
            clips.append((sec(b), sec(e)))
    assert clips and all(e > b for b, e in clips)
    assert all(clips[k][0] >= clips[k - 1][1] - 1e-3 for k in range(1, len(clips)))
    assert clips[-1][1] <= dur + 0.05
    for x in glob.glob(str(oebps / "p*.xhtml")):
        for v in re.findall(r"(?:left|top|width|height):(-?[\d.]+)%", open(x).read()):
            assert -2 <= float(v) <= 102

    if EPUBCHECK:
        r = subprocess.run(["java", "-jar", EPUBCHECK, out], capture_output=True, text=True)
        assert "0 errors / 0 warnings" in r.stdout + r.stderr, r.stdout + r.stderr


def test_image_model_arrow_is_printed_equals():
    from pdfaudiobook.math import reconcile_relations

    assert reconcile_relations(r"x\longrightarrow\frac{1}{2}", "x=12") == r"x=\frac{1}{2}"
    assert reconcile_relations(r"f\to g", "f→g") == r"f\to g"  # a real arrow stays


def test_word_subscripts_read_as_words():
    from pdfaudiobook.math import _word_subscripts

    assert _word_subscripts(r"I_{ave}+V_{o u t}+x_{0}+a_{ij}") == r"I_{\text{ave}}+V_{\text{out}}+x_{0}+a_{ij}"


def test_spoken_math_cleanup():
    from pdfaudiobook.math import _connectives, looks_like_latex

    t = _connectives(r"V_{AB}=\frac{U}{q} orV_{B}=1")
    assert r"\text{ or }V_{B}" in t
    assert "delta" in _connectives(r"\frac{∆Q}{\Delta t}") and "Delta" not in _connectives(r"\Delta t")
    assert _connectives(r"\textstyle\mathrm{I}=\operatorname*{lim}_{t\to0}x") == r"{I}=\text{the limit as }t\text{ approaches }0\text{ of }x"
    assert _connectives(r"\operatorname{for} A") == r"\operatorname{for} A"
    assert not looks_like_latex(r"\frac{2.8\times10^{.8}}{2}\longrightarrow\frac{3}{4}\longrightarrow")


def test_equation_number_and_table():
    from pdfaudiobook import mathlayout as ml

    assert ml.EQ_NUM_RE.match("(2.1)") and ml.EQ_NUM_RE.match("(3.4a)") and not ml.EQ_NUM_RE.match("(x)")


def test_line_offsets_match_paragraph_text():
    from pdfaudiobook.chapters import lines_to_paragraphs
    from pdfaudiobook.extract import Line

    ls = [Line("A long first line of a para-", 10, 0.10, bbox=(72, 100, 500, 112)),
          Line("graph that wraps onto this one.", 10, 0.115, bbox=(72, 113, 300, 125))]
    paras, boxes, _ = lines_to_paragraphs([(0, l) for l in ls])
    assert paras == ["A long first line of a paragraph that wraps onto this one."]
    (b1, b2), = boxes
    assert paras[0][b1[6]:b1[7]].startswith("A long") and paras[0][b2[6]:b2[7]] == "graph that wraps onto this one."
