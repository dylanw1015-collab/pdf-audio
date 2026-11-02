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
# A page number stuck to a running head: "152 Chapter 4 Gas Turbines", "xviii Preface", "Problems 103".
HEAD_NUM_RE = re.compile(r"^(\d{1,4}|[ivxlcdm]{1,6})\s+|\s+(\d{1,4}|[ivxlcdm]{1,6})$")
# Publisher boilerplate and figure credits nobody wants read aloud.
BOILERPLATE_RE = re.compile(
    r"©|\bcopyright\b|all rights reserved|\bisbn\b|(reproduced|reprinted) with permission|\bpublished \d{4} by\b",
    re.IGNORECASE,
)
# The credit inside a caption: "Source: Reproduced with permission from X." (the rest of the caption stays).
CREDIT_RE = re.compile(r"\s*Source:[^\[\]]*?(?:reproduced|reprinted)\s+with\s+permission[^\[\].]*(?:\.|(?=\[)|$)", re.IGNORECASE)
# "Figure 1.12", "Figures 1.12 and 1.13", "Table 3.2", "Fig. 2.4(a)": what a body paragraph points at.
REF_RE = re.compile(r"\b(Figures?|Figs?\.?|Tables?)\s+(P?\d+\.\d+)[a-z]?", re.IGNORECASE)
REF_TAIL_RE = re.compile(r"\s*(?:\([a-z]\))?\s*(?:,|and|&|or)\s*(P?\d+\.\d+)[a-z]?", re.IGNORECASE)
LABEL_ONLY_RE = re.compile(r"(figure|table)\s+P?\d+\.\d+\.?", re.IGNORECASE)
# "Chapter 3" alone on a line opens a chapter; its number is not a page number
# even when chapter n happens to start on page n.
OPENER_RE = re.compile(r"^\s*(chapter|part|appendix|section|lecture|unit)\s+(\d{1,3}|[ivxlc]{1,6})\s*$", re.IGNORECASE)
NUMBER_ONLY_RE = re.compile(r"^[\d\s.,\-–()]{1,9}$")  # stray page numbers and axis labels


@dataclass
class Chapter:
    title: str
    start_page: int  # 0-based
    end_page: int  # inclusive
    paragraphs: list[str] = field(default_factory=list)
    # Per paragraph, one [page, x0, y0, x1, y1, nchars] per source line (PDF points);
    # a zero box means the position is unknown (OCR of a turned scan). Two more
    # numbers follow: where the line's text starts and ends in the paragraph.
    boxes: list = field(default_factory=list)

    @property
    def geometry(self) -> list:
        """Per paragraph, [page, x0, y0, x1, y1, start, end] per line, page -1
        where the position is unknown (what the read-along EPUB highlights)."""
        return [[[b[0] if b[3] > b[1] else -1, *b[1:5], b[6], b[7]] for b in para if len(b) >= 8]
                for para in self.boxes]

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

    A running head that carries its page number ("152 Chapter 4 ...") is never
    exactly the same twice, and it repeats only within its chapter, so a second
    count ignores the page number and needs far fewer repeats.
    """
    def head_key(text: str) -> tuple[str, bool]:
        """Running-head text without its page number or math markup, and whether a number was removed."""
        t = re.sub(r"\s+", " ", re.sub(r"[$■]", "", text)).strip()  # math detection wraps section numbers in $..$
        s = t
        for _ in range(2):  # a number at the start and/or the end
            s = HEAD_NUM_RE.sub("", s)
        k = _norm(s)
        if k in ("chapter", "part", "appendix", "section", "page") or not re.search(r"[a-z]{2}", k):
            return "", False  # a bare "Chapter 4" opener, not a running head
        return k, s != t

    def in_band(l: Line) -> bool:
        return l.y0 < band or l.y0 > 1 - band

    counts: Counter[str] = Counter()
    folded: Counter[str] = Counter()
    for p in pages:
        counts.update(k for k in {_norm(l.text) for l in p.lines if in_band(l)} if k)
        folded.update(k for k, num in {head_key(l.text) for l in p.lines if in_band(l)} if k and num)
    min_repeat = max(3, int(0.25 * len(pages)))

    # A printed page number is the PDF page plus a fixed offset ("23" on PDF page 56). A margin line that
    # starts or ends with that number is a running head however seldom its words repeat.
    def number_in(text: str) -> int | None:
        m = re.search(r"^(\d{1,4})\s+\S|\S\s+(\d{1,4})$|^(\d{1,4})$", re.sub(r"[$■]", "", text).strip())
        return int(next(g for g in m.groups() if g)) if m else None

    offsets: dict[int, list[int]] = {}
    for p in pages:
        offsets[p.number] = [n - p.number for n in (number_in(l.text) for l in p.lines if in_band(l)) if n is not None]

    def page_offset(pn: int) -> int | None:
        near = Counter(o for j in range(pn - 15, pn + 16) for o in offsets.get(j, []))
        top = near.most_common(1)
        return top[0][0] if top and top[0][1] >= 3 else None

    def is_head(l: Line) -> bool:
        k, num = head_key(l.text)
        if num and folded[k] >= 3:  # same words, a different page number each time
            return True
        # short, unnumbered, and repeated: "(Continued)"
        return len(l.text) < 40 and "$" not in l.text and counts[_norm(l.text)] >= 8

    # Ebooks converted from print often keep the print running head inline
    # ("10 PRACTICAL ELECTRONICS FOR" / "INVENTORS"), so after reflow it sits
    # mid-page. A short line whose letters repeat on 8%+ of pages anywhere is
    # that head; figure and table labels are kept.
    anywhere: Counter[str] = Counter()
    for p in pages:
        anywhere.update({_letters(l.text) for l in p.lines if len(l.text.split()) <= 5})
    min_any = max(5, int(0.08 * len(pages)))

    def inline_head(l: Line) -> bool:
        k = _letters(l.text)
        return (len(l.text.split()) <= 5 and len(k) >= 6 and not l.caption
                and not k.startswith(("figure", "table")) and anywhere[k] >= min_any)

    for p in pages:
        off = page_offset(p.number)
        p.lines = [
            l
            for l in p.lines
            if not (in_band(l) and (PAGE_NUM_RE.match(l.text) or counts[_norm(l.text)] >= min_repeat or is_head(l)
                                    or (off is not None and number_in(l.text) == p.number + off and len(l.text) < 90
                                        and not OPENER_RE.match(l.text))))
            and not inline_head(l)
        ]


def _letters(s: str) -> str:
    return re.sub(r"[\W\d_]+", "", s.lower())


def classify_blocks(pages: list[Page]) -> None:
    """Mark figure/table caption blocks and the small text beside them, in place.

    A block whose first line is a "Figure 1.11" / "Table 2.1" label is a caption. Other blocks set
    clearly smaller than the body text are asides (spec boxes, figure labels); an aside with fewer than
    four words is an axis tick or "(a)" and is dropped. Math blocks are left alone.
    """
    body = _body_size(pages)
    for page in pages:
        if page.source != "text":
            continue
        blocks: dict[int, list[Line]] = {}
        for l in page.lines:
            if l.block >= 0:
                blocks.setdefault(l.block, []).append(l)
        drop: set[int] = set()
        for lines in blocks.values():
            if lines[0].caption.startswith(("Figure", "Table")):
                for l in lines:
                    l.role = "caption"
                continue
            if any(l.math for l in lines):
                continue
            sizes = sorted(l.size for l in lines)
            if sizes[len(sizes) // 2] < 0.92 * body:
                if len(re.findall(r"[A-Za-z]{2,}", " ".join(l.text for l in lines))) < 4:
                    drop.update(id(l) for l in lines)
                else:
                    for l in lines:
                        l.role = "aside"
        if drop:
            page.lines = [l for l in page.lines if id(l) not in drop]


def _refs(text: str) -> list[str]:
    """Keys ("Figure 1.12", "Table 3.2") a paragraph points at, in the order it mentions them."""
    out: list[str] = []
    for m in REF_RE.finditer(text):
        kind = "Table" if m.group(1).lower().startswith("t") else "Figure"
        ids = [m.group(2)]
        tail = REF_TAIL_RE.match(text, m.end())
        while tail:
            ids.append(tail.group(1))
            tail = REF_TAIL_RE.match(text, tail.end())
        out += [f"{kind} {i.upper()}" for i in ids]
    return list(dict.fromkeys(out))


def reorder_floats(paras: list[str], boxes: list[list], meta: list[dict]) -> tuple[list[str], list[list], list[dict]]:
    """Read a figure or table caption (and the box beside it) right after the first paragraph that
    mentions it, instead of wherever it happens to sit on the page. A float is read once; one that no
    nearby paragraph mentions stays where it was."""
    # A bare "Figure 2.3." label left in the body next to the real caption would be read before the
    # figure is discussed; the caption carries the label, so drop the leftover.
    real = {m["key"] for m in meta if m["kind"] == "caption" and m["key"]}
    keep = [i for i, (t, m) in enumerate(zip(paras, meta))
            if not (m["key"] in real and m["kind"] != "caption" and re.fullmatch(r"(Figure|Table) P?\d+\.\d+\.?", t.strip()))]
    paras, boxes, meta = [paras[i] for i in keep], [boxes[i] for i in keep], [meta[i] for i in keep]
    n = len(paras)
    cap_of: dict[str, int] = {}
    for i, m in enumerate(meta):
        if m["kind"] == "caption" and m["key"] and m["key"] not in cap_of:
            cap_of[m["key"]] = i
    groups = {k: [c] for k, c in cap_of.items()}
    span = lambda i, pg: [(b[2], b[4]) for b in boxes[i] if b[0] == pg and b[3] > b[1]]
    for i, m in enumerate(meta):  # attach each aside to the caption beside or just below/above it
        if m["kind"] != "aside":
            continue
        a = span(i, m["page"])
        best, gap = None, 60.0
        for k, c in cap_of.items():
            cs = span(c, meta[c]["page"])
            if a and cs and meta[c]["page"] == m["page"]:
                g = max(0.0, max(min(y for y, _ in a), min(y for y, _ in cs)) - min(max(y for _, y in a), max(y for _, y in cs)))
                if g <= gap:
                    best, gap = k, g
        if best:
            groups[best].append(i)
    member = {i: k for k, ix in groups.items() for i in ix}
    first_ref: dict[str, int] = {}
    for i in range(n):
        if i in member:
            continue
        for k in _refs(paras[i]):
            # A mention before the caption moves it up only if close; one after it always holds the caption
            # back (up to 40 pages), so a figure is never read before the text that discusses it.
            if k in groups and k not in first_ref and -3 <= meta[i]["page"] - meta[groups[k][0]]["page"] <= 40:
                first_ref[k] = i
    order: list[int] = []
    for i in range(n):
        if i in member:
            if member[i] not in first_ref:
                order.append(i)  # nobody mentions it: leave it where it is
            continue
        order.append(i)
        for k in _refs(paras[i]):
            if first_ref.get(k) == i:
                order += sorted(groups[k], key=lambda j: (j != groups[k][0], j))
    if sorted(order) != list(range(n)):  # never lose or repeat text
        return paras, boxes, meta
    return [paras[i] for i in order], [boxes[i] for i in order], [meta[i] for i in order]


def lines_to_paragraphs(lines: list[tuple[int, Line]]) -> tuple[list[str], list[list], list[dict]]:
    """Rejoin wrapped (page, line) pairs into paragraphs and undo end-of-line
    hyphenation. Also returns each paragraph's line boxes (for the page view) and
    its kind: "caption", "aside" or "body", with the caption's key and first page."""
    paras: list[str] = []
    boxes: list[list] = []
    meta: list[dict] = []
    cur = ""
    cur_boxes: list = []
    cur_meta: dict = {}
    prev: Line | None = None
    prev_q = -1
    for q, l in lines:
        t = re.sub(r"\s+", " ", re.sub(r"\$■\$", "", l.text)).strip()  # a bullet the math pass wrapped in $..$
        t = re.sub(r"[�-]", "", t)  # glyphs the PDF font could not map: nothing sensible to say
        if "$" not in t:
            t = re.sub(r"(?<=[A-Za-z0-9])-\s(?=[A-Za-z0-9])", "-", t)  # "step- down" -> "step-down" (typeset gap after a hyphen)
        if not t:
            continue
        gap = prev is not None and (l.y0 - prev.y0) > 2.2 * max(prev.size, 1) / 792
        size_jump = prev is not None and max(l.size, prev.size) > 1.15 * min(l.size, prev.size)
        ends_sentence = cur.endswith((".", "!", "?", '"', "”", ":"))
        # A new caption/example label, a change between caption, aside and body text, or a jump to a
        # block beside the previous one (not above or below it) all start a new paragraph.
        side_by_side = (prev is not None and q == prev_q and l.block >= 0 and prev.block >= 0 and l.block != prev.block
                        and max(l.bbox[0], prev.bbox[0]) >= min(l.bbox[2], prev.bbox[2]))
        # Inside one caption or aside block the line-spacing rules do not apply: its label is set in a
        # different size and ends in a period, and a "Source:" line ends in a colon.
        in_float = (prev is not None and q == prev_q and l.role != "body" and l.role == prev.role
                    and l.block == prev.block and l.block >= 0 and not l.caption)
        new_para = cur and not in_float and (
            gap or size_jump or (ends_sentence and len(prev.text) < 45) or l.y0 < (prev.y0 if prev else 0)
            or l.caption or l.role != prev.role or side_by_side)
        if new_para:
            paras.append(cur)
            boxes.append(cur_boxes)
            meta.append(cur_meta)
            cur, cur_boxes = t, []
        elif l.role == "aside" and cur:
            cur = f"{cur}{t}" if cur.endswith("-") else f"{cur} {t}" if cur.endswith(":") else f"{cur}; {t}"
        elif cur.endswith("-") and t[:1].islower():
            cur = cur[:-1] + t
        else:
            cur = f"{cur} {t}" if cur else t
        start = len(cur) - len(t)
        if not cur_boxes:
            key = l.caption if l.caption.startswith(("Figure", "Table")) else ""
            cur_meta = {"kind": l.role, "key": key, "page": q}
        cur_boxes.append([q, *[round(v, 1) for v in l.bbox], len(t), start, len(cur)])
        prev, prev_q = l, q
    if cur:
        paras.append(cur)
        boxes.append(cur_boxes)
        meta.append(cur_meta)
    paras = [CREDIT_RE.sub("", t).strip() for t in paras]
    # A bare "Figure 1.1." with no caption text says nothing to a listener, so it is dropped as well.
    keep = [i for i, t in enumerate(paras)
            if t and not (NUMBER_ONLY_RE.match(t) or LABEL_ONLY_RE.fullmatch(t) or (len(t) < 300 and BOILERPLATE_RE.search(t)))]
    return [paras[i] for i in keep], [boxes[i] for i in keep], [meta[i] for i in keep]


SKIP_RE = re.compile(
    r"^\s*(cover( page)?|front cover|back cover|inside (front|back) cover|half[- ]?title( page)?|title( page)?|"
    r"copyright( page)?|(table of )?contents( \(.*\))?|index( \(.*\))?|also by .*|about the (book|authors?)|"
    r"isbn[\s\d-]*[\dX]?|\d[\d-]{8,}[\dX])\s*$",
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
    from .extract import open_doc

    toc = open_doc(path).get_toc(simple=True)
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
    classify_blocks(pages)
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
        paras, boxes, meta = lines_to_paragraphs(lines)
        paras, boxes, meta = reorder_floats(paras, boxes, meta)
        end = npno if nli > 0 else npno - 1
        chapters.append(Chapter(title, pno, max(pno, min(end, len(pages) - 1)), paras, boxes))
    # Drop near-empty front matter (cover image, blank pages).
    if chapters and chapters[0].title == "Front matter" and len(chapters[0].text) < 200:
        chapters.pop(0)
    return chapters, used
