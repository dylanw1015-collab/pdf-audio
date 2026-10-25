"""Split extracted pages into chapters and clean the text for speech.

Order of strategies, most to least reliable:
1. The PDF outline (bookmarks). The author or publisher wrote it, so it is
   ground truth when present (ISO 32000-1, section 12.3.3 "Document Outline").
2. Heading detection. Lines that say "Chapter 7" / "CHAPTER VII" / "Part Two",
   or lines set noticeably larger than the body text, mark chapter starts.
   Font-size ranking is the same signal pymupdf4llm's IdentifyHeaders uses to
   turn PDFs into Markdown headings.
3. Fixed-size page groups, so even a book with no structure is navigable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher

import pymupdf

from .extract import Line, Page

CHAPTER_RE = re.compile(
    r"^\s*(chapter|chap\.|part|book|section|prologue|epilogue|introduction|preface|foreword|afterword|appendix)\b"
    r"(\s+([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty[\w-]*|thirty[\w-]*|forty[\w-]*|fifty[\w-]*)\b)?",
    re.IGNORECASE,
)
PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?[\divxlc]{1,6}\s*$", re.IGNORECASE)


@dataclass
class Chapter:
    title: str
    start_page: int  # 0-based
    end_page: int  # inclusive
    paragraphs: list[str] = field(default_factory=list)
    # Per paragraph, one [page, x0, y0, x1, y1, nchars] per source line (PDF points);
    # a zero box means the position is unknown (OCR of a turned scan).
    boxes: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


# ---------------------------------------------------------------- cleaning


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def strip_headers_footers(pages: list[Page], band: float = 0.08) -> None:
    """Remove page numbers and running headers/footers in place.

    A line in the top or bottom `band` of the page whose exact text repeats on
    at least 25% of pages is a running head, not content. Digits are kept in
    the comparison so "Chapter 1", "Chapter 2"... openers are not mistaken
    for one repeated header.
    """
    counts: Counter[str] = Counter()
    for p in pages:
        seen = {_norm(l.text) for l in p.lines if l.y0 < band or l.y0 > 1 - band}
        counts.update(k for k in seen if k)
    min_repeat = max(3, int(0.25 * len(pages)))
    for p in pages:
        p.lines = [
            l
            for l in p.lines
            if not (
                (l.y0 < band or l.y0 > 1 - band)
                and (PAGE_NUM_RE.match(l.text) or counts[_norm(l.text)] >= min_repeat)
            )
        ]


def lines_to_paragraphs(lines: list[tuple[int, Line]]) -> tuple[list[str], list[list]]:
    """Rejoin wrapped (page, line) pairs into paragraphs and undo end-of-line
    hyphenation. Also returns each paragraph's line boxes, for the page view."""
    paras: list[str] = []
    boxes: list[list] = []
    cur = ""
    cur_boxes: list = []
    prev: Line | None = None
    for q, l in lines:
        t = re.sub(r"\s+", " ", l.text).strip()
        if not t:
            continue
        gap = prev is not None and (l.y0 - prev.y0) > 2.2 * max(prev.size, 1) / 792
        size_jump = prev is not None and max(l.size, prev.size) > 1.15 * min(l.size, prev.size)
        ends_sentence = cur.endswith((".", "!", "?", '"', "”", ":"))
        new_para = cur and (gap or size_jump or (ends_sentence and len(prev.text) < 45) or l.y0 < (prev.y0 if prev else 0))
        if new_para:
            paras.append(cur)
            boxes.append(cur_boxes)
            cur, cur_boxes = t, []
        elif cur.endswith("-") and t[:1].islower():
            cur = cur[:-1] + t
        else:
            cur = f"{cur} {t}" if cur else t
        cur_boxes.append([q, *l.bbox, len(t)])
        prev = l
    if cur:
        paras.append(cur)
        boxes.append(cur_boxes)
    return paras, boxes


SKIP_RE = re.compile(
    r"^\s*(cover( page)?|front cover|back cover|inside (front|back) cover|half[- ]?title( page)?|title page|"
    r"copyright( page)?|(table of )?contents|index|also by .*|\d[\d-]{8,}[\dX])\s*$",
    re.IGNORECASE,
)


def is_skippable(ch: Chapter) -> bool:
    """Sections nobody wants read aloud: covers, copyright, contents, index."""
    return bool(SKIP_RE.match(ch.title))


def spoken_text(ch: Chapter) -> str:
    """Chapter text for speech, announcing the title unless the text already
    opens with it (so the heading is not read twice)."""
    t = _norm(ch.title)
    head = _norm(" ".join(ch.paragraphs[:2]))[: len(t)]
    if ch.paragraphs and SequenceMatcher(None, t, head).ratio() > 0.8:
        return ch.text
    return f"{ch.title}.\n\n{ch.text}"


# ---------------------------------------------------------------- detection


def _body_size(pages: list[Page]) -> float:
    c: Counter[float] = Counter()
    for p in pages:
        for l in p.lines:
            c[round(l.size)] += len(l.text)
    return c.most_common(1)[0][0] if c else 10.0


def _find_line(page: Page, title: str) -> int:
    """Index of the line on `page` where `title` starts, or 0.

    Titles often span two lines ("Chapter Two" / "A Letter Arrives"), so each
    line is also scored joined with the next one; the earliest best wins.
    """
    target = _norm(title)
    lines = page.lines[:15]
    best, best_i = 0.0, 0
    for i, l in enumerate(lines):
        cands = [_norm(l.text)]
        if i + 1 < len(lines):
            cands.append(cands[0] + _norm(lines[i + 1].text))
        r = max(SequenceMatcher(None, target, c).ratio() for c in cands)
        if r > best + 1e-9:
            best, best_i = r, i
    return best_i if best >= 0.6 else 0


MAX_PAGES_PER_CHAPTER = 40  # ~2 h of audio for a typical textbook page


def _auto_level(toc: list, n_pages: int) -> int:
    """Shallowest bookmark depth whose entries are a navigable size.

    Some books bookmark Parts at the top and chapters one or two levels
    down. Going deeper until entries average at most 40 pages gives
    chapter-sized pieces while keeping section-level clutter out.
    """
    deepest = max(t[0] for t in toc)
    for lvl in range(1, deepest + 1):
        n = sum(1 for t in toc if t[0] <= lvl)
        if n >= 3 and n_pages / n <= MAX_PAGES_PER_CHAPTER:
            return lvl
    return 1 if sum(1 for t in toc if t[0] == 1) >= 2 else min(2, deepest)


def from_outline(path: str, pages: list[Page], level: int | None) -> list[tuple[str, int, int]]:
    """Chapter starts from the PDF bookmarks: (title, page, line_index)."""
    toc = pymupdf.open(path).get_toc(simple=True)
    if not toc:
        return []
    if level is None:
        level = _auto_level(toc, len(pages))
    starts = []
    for lvl, title, pno in toc:
        # Entries above the chosen depth are kept too (a Part's opening page,
        # a Preface with no subsections), so no text falls between chapters.
        if lvl <= level and 1 <= pno <= len(pages):
            p = pages[pno - 1]
            starts.append((title.strip(), pno - 1, _find_line(p, title)))
    return starts


def from_headings(pages: list[Page]) -> list[tuple[str, int, int]]:
    """Chapter starts guessed from heading text and font size."""
    body = _body_size(pages)
    keyword, big = [], []
    for p in pages:
        found_big = False
        for i, l in enumerate(p.lines):
            if l.math or len(l.text) > 60:
                continue
            # "Chapter 7" style headings can start anywhere on a page (a new
            # chapter often begins mid-page, and a scanned spread holds two
            # pages), so look at every short line for them.
            if CHAPTER_RE.match(l.text) and len(l.text.split()) <= 8:
                keyword.append((p.number, i, l))
                continue
            if not found_big and i < 8 and l.y0 <= 0.5 and l.size >= 1.3 * body and len(l.text.split()) <= 12:
                big.append((p.number, i, l))
                found_big = True
    # Running heads like "BOOK 1  THE CONFESSIONS  17" repeat with only the
    # numbers changing; a real heading has just the keyword and its number.
    shape = Counter(re.sub(r"[\d\W_]+", "", k[2].text.lower()) for k in keyword)
    keyword = [k for k in keyword
               if not (len(k[2].text.split()) >= 3 and shape[re.sub(r"[\d\W_]+", "", k[2].text.lower())] >= 2)]
    picked = keyword if len(keyword) >= 2 else []
    if not picked and big:
        # Pick the largest size tier that occurs at least twice; a single
        # largest line is usually the book's title page.
        tiers = Counter(round(l.size) for _, _, l in big)
        for size in sorted(tiers, reverse=True):
            if tiers[size] >= 2:
                picked = [b for b in big if round(b[2].size) >= size]
                break
    # A chapter-title running head repeats on every page of its chapter; keep
    # only its first appearance.
    deduped = []
    for item in picked:
        if not deduped or _norm(deduped[-1][2].text) != _norm(item[2].text):
            deduped.append(item)
    picked = deduped
    starts = []
    for pno, i, l in picked:
        title = l.text.strip()
        # "Chapter 3" followed by a subtitle line: join them.
        p = pages[pno]
        if i + 1 < len(p.lines):
            nxt = p.lines[i + 1]
            if nxt.size >= 1.15 * body and len(nxt.text) < 80 and len(title) < 20 and not nxt.math:
                title = f"{title}: {nxt.text.strip()}"
        starts.append((title, pno, i))
    return starts


def fixed_chunks(pages: list[Page], per: int = 15) -> list[tuple[str, int, int]]:
    return [(f"Pages {s + 1}-{min(s + per, len(pages))}", s, 0) for s in range(0, len(pages), per)]


def split_chapters(
    path: str,
    pages: list[Page],
    strategy: str = "auto",
    outline_level: int | None = None,
) -> tuple[list[Chapter], str]:
    """Return (chapters, strategy_used)."""
    strip_headers_footers(pages)
    starts: list[tuple[str, int, int]] = []
    used = strategy
    if strategy in ("auto", "outline"):
        starts, used = from_outline(path, pages, outline_level), "outline"
    if not starts and strategy in ("auto", "headings"):
        starts, used = from_headings(pages), "headings"
    if not starts:
        starts, used = fixed_chunks(pages), "pages"

    starts.sort(key=lambda s: (s[1], s[2]))
    # Anything before the first chapter becomes "Front matter" if it has text.
    if starts[0][1] > 0 or starts[0][2] > 0:
        starts.insert(0, ("Front matter", 0, 0))

    chapters = []
    for k, (title, pno, li) in enumerate(starts):
        npno, nli = (starts[k + 1][1], starts[k + 1][2]) if k + 1 < len(starts) else (len(pages), 0)
        lines: list[tuple[int, Line]] = []
        for q in range(pno, min(npno, len(pages) - 1) + 1):
            pl = pages[q].lines
            lo = li if q == pno else 0
            hi = nli if q == npno else len(pl)
            # A turned scan's boxes are in the turned image's space: mark them unknown.
            lines.extend((q, replace(l, bbox=(0, 0, 0, 0)) if pages[q].rotated else l) for l in pl[lo:hi])
        paras, boxes = lines_to_paragraphs(lines)
        end = npno if nli > 0 else npno - 1
        chapters.append(Chapter(title, pno, max(pno, min(end, len(pages) - 1)), paras, boxes))
    # Drop near-empty front matter (cover image, blank pages).
    if chapters and chapters[0].title == "Front matter" and len(chapters[0].text) < 200:
        chapters.pop(0)
    return chapters, used
