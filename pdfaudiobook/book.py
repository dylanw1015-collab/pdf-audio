"""Load a PDF into chapters, caching the result so OCR runs only once per book."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

from .chapters import Chapter, split_chapters
from .extract import extract_pages

from . import __version__

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pdfaudiobook", "cache")


def _key(path: str, **opts) -> str:
    st = os.stat(path)
    # The version is part of the key so an upgrade re-reads books with the
    # improved extraction instead of serving old cached text.
    raw = f"{__version__}|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{sorted(opts.items())}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_book(
    path: str,
    mode: str = "auto",
    strategy: str = "auto",
    outline_level: int | None = None,
    lang: str = "eng",
    use_cache: bool = True,
    progress=None,
    math: bool = True,
) -> dict:
    """Return {"title", "strategy", "ocr_pages", "chapters": [Chapter...]}."""
    key = _key(path, mode=mode, strategy=strategy, level=outline_level, lang=lang, math=math)
    cache = os.path.join(CACHE_DIR, key + ".json")
    if use_cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            d = json.load(f)
        d["chapters"] = [Chapter(**c) for c in d["chapters"]]
        return d

    pages = extract_pages(path, mode=mode, lang=lang, progress=progress, math=math)
    chapters, used = split_chapters(path, pages, strategy, outline_level)
    d = {
        "title": os.path.splitext(os.path.basename(path))[0],
        "strategy": used,
        "ocr_pages": [p.number + 1 for p in pages if p.source == "ocr"],
        "chapters": chapters,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({**d, "chapters": [asdict(c) for c in chapters]}, f, ensure_ascii=False)
    return d
