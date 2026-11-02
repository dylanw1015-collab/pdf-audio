"""Read-along EPUB: the original printed pages, with each sentence highlighted
as the narration reaches it.

Why this format
* EPUB 3 Media Overlays pair text with recorded audio through SMIL files,
  one <par> per sentence (W3C, EPUB Media Overlays 3.2/3.3,
  https://w3c.github.io/epub-specs/archive/epub32/spec/epub-mediaoverlays.html).
  The reading system adds the class named by `media:active-class` to the
  element being read, so a CSS rule draws the highlight.
* Apple Books plays Media Overlays ("Read Aloud") but "only in fixed-layout
  books" (Apple Books Asset Guide,
  https://help.apple.com/itc/booksassetguide/en.lproj/itc6db639756.html).
  So every page is a fixed page holding the PDF page as an image, which also
  keeps figures and equations exactly as printed.
* Highlights are boxes over the page image. Boxes come from the line geometry
  kept during extraction (text layer or Tesseract). Where a sentence starts or
  ends inside a line, the x position is interpolated by character count, which
  is close for proportional text but can be off by a few letters.
* Audio is AAC in an MP4 (.m4a), a core EPUB audio type and the format in
  Apple's own Read Aloud samples.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
import zipfile
from datetime import datetime, timezone

import pymupdf

from .chapters import is_skippable
from .export import _ffmpeg, safe_name
from .math import DISPLAY_RE, MathSpeaker

_SENT_RE = re.compile(r"(?<=[.!?”\"])\s+(?=[A-Z“\"'(])")
GAP_S = 0.35  # pause after each sentence, as in tts.concat_wavs

CSS = """
html, body { margin: 0; padding: 0; }
.page { position: relative; }
.page img { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
.s b { position: absolute; display: block; }
/* Multiply keeps the printed ink dark under the colour, like a real highlighter. */
.-epub-media-overlay-active b { background-color: #ffe14d; mix-blend-mode: multiply; opacity: 0.85; }
"""


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence; display equations are their own unit."""
    out = []
    pos = 0
    for m in list(DISPLAY_RE.finditer(text)) + [None]:
        stop = m.start() if m else len(text)
        seg = text[pos:stop]
        k = 0
        for part in _SENT_RE.split(seg):
            i = seg.index(part, k)
            k = i + len(part)
            a, b = pos + i, pos + k
            if text[a:b].strip():
                out.append((a, b))
        if m:
            out.append((m.start(), m.end()))
            pos = m.end()
    return out


def boxes_for(geo: list, a: int, b: int) -> dict[int, list[tuple]]:
    """Page -> boxes (PDF points) covering characters a..b of a paragraph."""
    out: dict[int, list[tuple]] = {}
    for page, x0, y0, x1, y1, ls, le in geo:
        lo, hi = max(a, ls), min(b, le)
        if page < 0 or hi <= lo:
            continue
        n = max(le - ls, 1)
        out.setdefault(page, []).append(
            (x0 + (x1 - x0) * (lo - ls) / n, y0, x0 + (x1 - x0) * (hi - ls) / n, y1))
    return out


def _clock(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:06.3f}"


def _wav_seconds(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def export_epub(pdf: str, book: dict, engine, out_dir: str, chapters=None,
                width_px: int = 1100, jpeg_quality: int = 70, crop: bool = False, strips: int = 1,
                log=print) -> str:
    """Write <title>.epub with every page of `pdf` and narration for `chapters`
    (1-based numbers; default: all except cover/contents/index).

    crop: trim the white margins so the printed text fills more of a phone
    screen. An iPhone 15 screen is 1179 px at 460 ppi = 2.56 in wide (Apple
    tech specs), so a whole 8.5 in Letter page shows at about 30% of its
    printed size; trimming 1 in margins each side raises that to about 40%.

    strips: cut each page into this many horizontal strips, one per EPUB
    page, for reading with the phone turned sideways. In landscape the screen
    is 2.56 in tall and about 5.5 in wide, so a third of a 6.5 x 9 in text
    block (6.5 x 3 in) shows at about 85% of printed size. Cuts are moved to
    the nearest blank row so no line of text is sliced."""
    os.makedirs(out_dir, exist_ok=True)
    from .extract import open_doc

    doc = open_doc(pdf)
    chs = book["chapters"]
    idx = chapters or [i for i in range(1, len(chs) + 1) if not is_skippable(chs[i - 1])]
    speaker = MathSpeaker()
    # page -> list of (sentence id, boxes, audio file, clip begin, clip end)
    pars: dict[int, list] = {}
    audio_items, durations = [], {}
    work = tempfile.mkdtemp(prefix="pdfab_epub_")
    root = os.path.join(work, "book")
    os.makedirs(os.path.join(root, "OEBPS", "audio"))
    os.makedirs(os.path.join(root, "OEBPS", "images"))
    os.makedirs(os.path.join(root, "META-INF"))
    sid = 0
    try:
        for n in idx:
            ch = chs[n - 1]
            log(f"[{n}/{len(chs)}] {ch.title}")
            parts, t = [], 0.0
            for para, geo in zip(ch.paragraphs, ch.geometry or [[]] * len(ch.paragraphs)):
                for a, b in sentences(para):
                    spoken = speaker.speak_text(para[a:b]).strip()
                    by_page = boxes_for(geo, a, b)
                    if not re.search(r"\w", spoken) or not by_page:
                        continue  # nothing to say, or nowhere on the page to show it
                    wav = os.path.join(work, f"{len(parts):05d}.wav")
                    engine.to_wav(spoken, wav)
                    dur = _wav_seconds(wav)
                    parts.append(wav)
                    # A sentence that runs onto the next page is split there,
                    # with its audio divided by how much text is on each page.
                    total = sum(sum(x1 - x0 for x0, _, x1, _ in bx) for bx in by_page.values()) or 1
                    t0 = t
                    for k, (pg, bx) in enumerate(sorted(by_page.items())):
                        share = sum(x1 - x0 for x0, _, x1, _ in bx) / total
                        t1 = t + dur if k == len(by_page) - 1 else t0 + dur * share
                        sid += 1
                        pars.setdefault(pg, []).append([f"s{sid}", bx, f"ch{n:03d}.m4a", t0, t1])
                        t0 = t1
                    t += dur + GAP_S
            if not parts:
                continue
            joined = os.path.join(work, f"ch{n:03d}.wav")
            _concat(parts, joined)
            m4a = os.path.join(root, "OEBPS", "audio", f"ch{n:03d}.m4a")
            subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", joined, "-ac", "1",
                            "-c:a", "aac", "-b:a", "48k", m4a], check=True)
            audio_items.append(f"ch{n:03d}.m4a")
            for p in parts + [joined]:
                os.remove(p)

        log("Rendering pages...")
        views = []  # (view id, image file, page number)
        first_view = {}
        book_box = _book_box(doc) if crop else None
        # Only the pages the chosen chapters span, so one section of a big
        # book makes a small file.
        first = min(chs[n - 1].start_page for n in idx)
        last = max(chs[n - 1].end_page for n in idx)
        for i in range(first, last + 1):
            pg = doc[i]
            clip = pg.rect
            if crop:
                c = _content_box(pg)
                clip = (book_box | c if c else book_box) & pg.rect
            for k, band in enumerate(_strips(pg, clip, strips)):
                vid = f"p{i + 1:04d}" + ("abcdefgh"[k] if strips > 1 else "")
                first_view.setdefault(i, vid)
                zoom = width_px / band.width
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=band)
                img = vid + ".jpg"
                pix.save(os.path.join(root, "OEBPS", "images", img), jpg_quality=jpeg_quality)
                views.append((vid, img, i))
                ps = _pars_in(pars.get(i, []), band)
                with open(os.path.join(root, "OEBPS", vid + ".xhtml"), "w", encoding="utf-8") as f:
                    f.write(_page_xhtml(vid, i, img, pix.width, pix.height, band, ps))
                if ps:
                    with open(os.path.join(root, "OEBPS", vid + ".smil"), "w", encoding="utf-8") as f:
                        f.write(_smil(vid, ps))
                    durations[vid] = sum(p[4] - p[3] for p in ps)

        toc = [(chs[n - 1].title, first_view[chs[n - 1].start_page]) for n in range(1, len(chs) + 1)
               if chs[n - 1].start_page in first_view]
        _write_package(root, book["title"], views, durations, audio_items, toc)
        out = os.path.join(out_dir, safe_name(book["title"]) + ".epub")
        _zip(root, out)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _ink_rows(pg, clip):
    """Per-row "has ink" flags for `clip` at 72 dpi (one row per point)."""
    import numpy as np

    pix = pg.get_pixmap(clip=clip, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    return (a < 200).any(axis=1)


def _strips(pg, clip, n: int) -> list:
    if n <= 1:
        return [clip]
    ink = _ink_rows(pg, clip)
    cuts = [clip.y0]
    for k in range(1, n):
        want = int(len(ink) * k / n)
        # nearest blank row within a quarter strip of the ideal cut
        reach = max(len(ink) // (4 * n), 1)
        best = min((r for r in range(max(want - reach, 0), min(want + reach, len(ink))) if not ink[r]),
                   key=lambda r: abs(r - want), default=want)
        cuts.append(clip.y0 + best * clip.height / max(len(ink), 1))
    cuts.append(clip.y1)
    return [pymupdf.Rect(clip.x0, a, clip.x1, b) for a, b in zip(cuts, cuts[1:]) if b - a > 1]


def _pars_in(ps: list, band) -> list:
    """The sentence pieces of a page that fall in `band`. A sentence whose
    boxes straddle two strips is split, its audio divided by box width."""
    out = []
    for s, boxes, a, t0, t1 in ps:
        inside = [b for b in boxes if band.y0 <= (b[1] + b[3]) / 2 < band.y1]
        if not inside:
            continue
        if len(inside) == len(boxes):
            out.append([s, boxes, a, t0, t1])
            continue
        w = lambda bs: sum(x1 - x0 for x0, _, x1, _ in bs)
        before = [b for b in boxes if (b[1] + b[3]) / 2 < band.y0]
        total = w(boxes) or 1
        b0 = t0 + (t1 - t0) * w(before) / total
        b1 = b0 + (t1 - t0) * w(inside) / total
        out.append([f"{s}_{int(band.y0)}", inside, a, b0, b1])
    return out


def _content_box(pg) -> pymupdf.Rect | None:
    """Bounding box of everything inked on the page, from a 36 dpi render."""
    import numpy as np

    pix = pg.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5), colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    ys, xs = np.nonzero(a < 200)
    if not len(xs):
        return None
    pad = 6  # points of breathing room around the ink
    r = pymupdf.Rect(xs.min() * 2 - pad, ys.min() * 2 - pad, xs.max() * 2 + 2 + pad, ys.max() * 2 + 2 + pad)
    return r + (pg.rect.x0, pg.rect.y0, pg.rect.x0, pg.rect.y0)


def _book_box(doc) -> pymupdf.Rect:
    """One text-block box for the whole book, so the zoom stays the same from
    page to page; a page with ink outside it is widened to fit (never cut).
    Uses the median edges of pages 10%-90% of the way through (skips covers)."""
    import statistics

    n = len(doc)
    step = max(1, n // 200)  # a 200-page sample is plenty for a median
    boxes = [b for b in (_content_box(doc[i]) for i in range(n // 10, max(n - n // 10, 1), step)) if b]
    if not boxes:
        return doc[0].rect
    med = lambda k: statistics.median(b[k] for b in boxes)
    return pymupdf.Rect(med(0), med(1), med(2), med(3))


def _concat(parts: list[str], out: str) -> None:
    with wave.open(parts[0]) as w:
        params = w.getparams()
    silence = b"\x00" * int(params.framerate * GAP_S) * params.sampwidth * params.nchannels
    with wave.open(out, "wb") as o:
        o.setparams(params)
        for p in parts:
            with wave.open(p) as w:
                o.writeframes(w.readframes(w.getnframes()) + silence)


def _page_xhtml(vid: str, i: int, img: str, w: int, h: int, rect, ps: list) -> str:
    spans = []
    for s, boxes, *_ in ps:
        bs = "".join(
            f'<b style="left:{100 * (x0 - 1 - rect.x0) / rect.width:.2f}%;top:{100 * (y0 - 1 - rect.y0) / rect.height:.2f}%;'
            f'width:{100 * (x1 - x0 + 2) / rect.width:.2f}%;height:{100 * (y1 - y0 + 2) / rect.height:.2f}%"></b>'
            for x0, y0, x1, y1 in boxes)
        spans.append(f'<span class="s" id="{s}">{bs}</span>')
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head><meta charset="UTF-8"/><title>Page {i + 1}</title>
<meta name="viewport" content="width={w}, height={h}"/>
<link rel="stylesheet" href="style.css"/></head>
<body><div class="page" style="width:{w}px;height:{h}px">
<img src="images/{img}" alt="Page {i + 1}"/>
{"".join(spans)}
</div></body></html>
"""


def _smil(vid: str, ps: list) -> str:
    body = "\n".join(
        f'<par id="par_{s}"><text src="{vid}.xhtml#{s}"/>'
        f'<audio src="audio/{a}" clipBegin="{_clock(b)}" clipEnd="{_clock(e)}"/></par>'
        for s, _, a, b, e in ps)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
<body><seq id="seq1" epub:textref="{vid}.xhtml">
{body}
</seq></body></smil>
"""


def _write_package(root: str, title: str, pages: list, durations: dict, audio: list, toc: list) -> None:
    t = html.escape(title)
    with open(os.path.join(root, "mimetype"), "w") as f:
        f.write("application/epub+zip")
    with open(os.path.join(root, "META-INF", "container.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                '<rootfiles><rootfile full-path="OEBPS/package.opf" media-type="application/oebps-package+xml"/>'
                "</rootfiles></container>\n")
    # Older Apple Books builds look for this file to switch on fixed layout.
    with open(os.path.join(root, "META-INF", "com.apple.ibooks.display-options.xml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<display_options><platform name="*">'
                '<option name="fixed-layout">true</option><option name="open-to-spread">false</option>'
                "</platform></display_options>\n")
    with open(os.path.join(root, "OEBPS", "style.css"), "w") as f:
        f.write(CSS)
    nav = "\n".join(f'<li><a href="{v}.xhtml">{html.escape(c)}</a></li>' for c, v in toc)
    with open(os.path.join(root, "OEBPS", "nav.xhtml"), "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
<head><meta charset="UTF-8"/><title>{t}</title><meta name="viewport" content="width=1100, height=1400"/></head>
<body><nav epub:type="toc" id="toc"><h1>Contents</h1><ol>
{nav}
</ol></nav></body></html>
""")
    items, spine, meta = [], [], []
    items.append('<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')
    items.append('<item id="css" href="style.css" media-type="text/css"/>')
    for k, (pid, img, _) in enumerate(pages):
        mo = f' media-overlay="mo_{pid}"' if pid in durations else ""
        items.append(f'<item id="{pid}" href="{pid}.xhtml" media-type="application/xhtml+xml"{mo}/>')
        props = ' properties="cover-image"' if k == 0 else ""
        items.append(f'<item id="img_{pid}" href="images/{img}" media-type="image/jpeg"{props}/>')
        if pid in durations:
            items.append(f'<item id="mo_{pid}" href="{pid}.smil" media-type="application/smil+xml"/>')
            meta.append(f'<meta property="media:duration" refines="#mo_{pid}">{_clock(durations[pid])}</meta>')
        spine.append(f'<itemref idref="{pid}"/>')
    for k, a in enumerate(audio):
        items.append(f'<item id="audio{k}" href="audio/{a}" media-type="audio/mp4"/>')
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid" xml:lang="en"
 prefix="rendition: http://www.idpf.org/vocab/rendition/#">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="uid">urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, title)}</dc:identifier>
<dc:title>{t}</dc:title>
<dc:language>en</dc:language>
<meta property="dcterms:modified">{now}</meta>
<meta property="rendition:layout">pre-paginated</meta>
<meta property="rendition:spread">none</meta>
<meta property="media:active-class">-epub-media-overlay-active</meta>
<meta property="media:duration">{_clock(sum(durations.values()))}</meta>
{chr(10).join(meta)}
</metadata>
<manifest>
{chr(10).join(items)}
</manifest>
<spine>
{chr(10).join(spine)}
</spine>
</package>
"""
    with open(os.path.join(root, "OEBPS", "package.opf"), "w", encoding="utf-8") as f:
        f.write(opf)


def _zip(root: str, out: str) -> None:
    """The mimetype entry must come first and be stored uncompressed (EPUB OCF)."""
    with zipfile.ZipFile(out, "w") as z:
        z.write(os.path.join(root, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
        for d, _, files in os.walk(root):
            for fn in sorted(files):
                p = os.path.join(d, fn)
                rel = os.path.relpath(p, root)
                if rel != "mimetype":
                    z.write(p, rel, compress_type=zipfile.ZIP_DEFLATED)
