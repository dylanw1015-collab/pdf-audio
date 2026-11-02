"""Command line: list chapters, read aloud, export audio, dump text."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from .book import load_book
from .export import export_book
from .chapters import is_skippable, spoken_text
from .math import MathOCR, MathSpeaker
from .tts import chunk_text, get_engine, play_wav

PROGRESS_FILE = os.path.join(os.path.expanduser("~"), ".pdfaudiobook", "progress.json")


def _progress(i, n, page):
    tag = "OCR " if page.source == "ocr" else "text"
    print(f"\r  page {i}/{n} [{tag} q={page.quality:.2f}]", end="", file=sys.stderr, flush=True)
    if i == n:
        print(file=sys.stderr)


def _load(a):
    if a.fast_math:
        MathOCR.disabled = True
    return load_book(a.pdf, mode=a.ocr, strategy=a.chapters, outline_level=a.level,
                     lang=a.lang, use_cache=not a.no_cache, progress=_progress, math=not a.no_math)


def _parse_range(s: str | None, n: int, chapters=None, include_all: bool = True) -> list[int]:
    if not s:
        return [i for i in range(1, n + 1) if include_all or not is_skippable(chapters[i - 1])]
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return [i for i in out if 1 <= i <= n]


def cmd_chapters(a):
    b = _load(a)
    print(f"{b['title']}  (chapters from: {b['strategy']}; OCR pages: {len(b['ocr_pages'])})")
    for i, c in enumerate(b["chapters"], 1):
        words = len(c.text.split())
        skip = "  (skipped)" if is_skippable(c) and not a.all else ""
        print(f"{i:3d}. {c.title[:60]:60s} p{c.start_page + 1}-{c.end_page + 1}  {words:6d} words  ~{words / 155:4.0f} min{skip}")


def cmd_text(a):
    b = _load(a)
    speaker = MathSpeaker() if a.spoken else None
    for i in _parse_range(a.only, len(b["chapters"])):
        c = b["chapters"][i - 1]
        body = speaker.speak_text(c.text) if speaker else c.text
        print(f"\n===== {i}. {c.title} =====\n\n{body}")


def cmd_export(a):
    b = _load(a)
    eng = get_engine(a.engine, a.voice, a.rate)
    print(f"Using TTS engine: {eng.name}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.pdf)), b["title"] + "_audio")
    if a.format == "epub":
        from .epub import export_epub

        only = _parse_range(a.only, len(b["chapters"]), b["chapters"], a.all) if (a.only or a.all) else None
        print("Wrote:\n  " + export_epub(a.pdf, b, eng, out, only, width_px=a.page_width,
                                                    crop=a.crop, strips=a.strips))
        return
    files = export_book(b, eng, out, a.format, _parse_range(a.only, len(b["chapters"]), b["chapters"], a.all))
    print("Wrote:\n  " + "\n  ".join(files))


def cmd_read(a):
    b = _load(a)
    eng = get_engine(a.engine, a.voice, a.rate)
    key = os.path.abspath(a.pdf)
    try:
        with open(PROGRESS_FILE) as f:
            prog = json.load(f)
    except (OSError, ValueError):
        prog = {}
    start_ch, start_chunk = (a.chapter, 0) if a.chapter else tuple(prog.get(key, (1, 0)))
    speaker = MathSpeaker()
    print(f"Engine: {eng.name}, math speech: {speaker.engine}. Ctrl+C stops and remembers your place.")
    tmp = tempfile.mkdtemp(prefix="pdfab_read_")
    try:
        for n in range(start_ch, len(b["chapters"]) + 1):
            ch = b["chapters"][n - 1]
            if is_skippable(ch) and not a.all and n != a.chapter:
                continue
            chunks = chunk_text(speaker.speak_text(spoken_text(ch)))
            print(f"\n== Chapter {n}: {ch.title} ==")
            for k in range(start_chunk if n == start_ch else 0, len(chunks)):
                prog[key] = (n, k)
                print("  " + chunks[k][:100] + ("..." if len(chunks[k]) > 100 else ""))
                wav = os.path.join(tmp, "now.wav")
                eng.to_wav(chunks[k], wav)
                play_wav(wav)
        prog.pop(key, None)
    except KeyboardInterrupt:
        print(f"\nStopped at chapter {prog[key][0]}. Run the same command to resume.")
    finally:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(prog, f)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pdfaudiobook", description="Turn PDF books into chaptered audio.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("pdf")
    common.add_argument("--ocr", choices=["auto", "text", "ocr"], default="auto",
                        help="auto: OCR only pages whose text looks garbled (default)")
    common.add_argument("--chapters", choices=["auto", "outline", "headings", "pages"], default="auto")
    common.add_argument("--level", type=int, help="bookmark level to treat as chapters (1 = top)")
    common.add_argument("--lang", default="eng", help="Tesseract language code(s), e.g. eng+fra")
    common.add_argument("--no-cache", action="store_true")
    common.add_argument("--no-math", action="store_true", help="skip equation detection")
    common.add_argument("--all", action="store_true", help="also read cover, copyright, contents and index")
    common.add_argument("--fast-math", action="store_true",
                        help="skip the image model for matrices/complex layouts (says 'an equation appears here')")
    tts = argparse.ArgumentParser(add_help=False)
    tts.add_argument("--engine", choices=["auto", "piper", "system", "espeak"], default="auto")
    tts.add_argument("--voice", help="piper voice name (e.g. en_US-ryan-high) or system voice id")
    tts.add_argument("--rate", type=float, default=1.0, help="speed multiplier, e.g. 1.25")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chapters", parents=[common], help="list detected chapters").set_defaults(f=cmd_chapters)
    t = sub.add_parser("text", parents=[common], help="print cleaned chapter text")
    t.add_argument("--only", help="chapters to print, e.g. 3 or 2-5,8")
    t.add_argument("--spoken", action="store_true", help="show equations as they will be read aloud")
    t.set_defaults(f=cmd_text)
    r = sub.add_parser("read", parents=[common, tts], help="read aloud, resuming where you stopped")
    r.add_argument("--chapter", type=int, help="start at this chapter")
    r.set_defaults(f=cmd_read)
    e = sub.add_parser("export", parents=[common, tts], help="write audio files per chapter")
    e.add_argument("--format", choices=["mp3", "m4b", "mp3+m4b", "wav", "epub"], default="mp3+m4b",
                   help="epub: the original pages with read-along highlighting (Apple Books Read Aloud)")
    e.add_argument("--page-width", type=int, default=1100, help="epub page image width in pixels")
    e.add_argument("--strips", type=int, default=1,
                   help="epub: cut each page into N strips for reading with the phone sideways (3 is ~85%% of print size)")
    e.add_argument("--crop", action="store_true", help="epub: trim the white page margins (default: show the whole page; pinch to zoom)")
    e.add_argument("--only", help="chapters to export, e.g. 1-3")
    e.add_argument("-o", "--out", help="output folder")
    e.set_defaults(f=cmd_export)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
