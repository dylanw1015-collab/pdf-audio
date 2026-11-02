# pdfaudiobook

Turns PDF books into audiobooks you can navigate by chapter, including PDFs where copy-paste gives junk (broken or custom fonts), scanned books with no text at all, and books with equations.

**Desktop window:** run `pdfaudiobook-gui` (or `pdfaudiobook-gui book.pdf`). Open a PDF, click a chapter, press Play. The text pane shows the exact words being spoken, equations included, with the current passage highlighted; click any passage to jump there. Space plays/pauses, ← and → skip a passage, the slider sets the speed, and *Export audiobook…* writes the MP3s and M4B. The window remembers where you stopped in each book (shared with the `read` command).

**Command line:**

```
pdfaudiobook chapters  book.pdf              # see the detected chapters
pdfaudiobook read      book.pdf              # read aloud now; Ctrl+C saves your place
pdfaudiobook read      book.pdf --chapter 5  # jump to a chapter
pdfaudiobook export    book.pdf              # MP3 per chapter + one .m4b with chapter markers
pdfaudiobook text      book.pdf --only 3 --spoken   # check what will be said, equations included
```

## On your iPhone: original pages, highlighted as they're read

```
pdfaudiobook export book.pdf --format epub              # whole pages, pinch to zoom
pdfaudiobook export book.pdf --format epub --rate 1.5   # narration 1.5x faster
pdfaudiobook export book.pdf --format epub --strips 3   # 3 strips per page, for reading sideways
```

Or click *Export for iPhone…* in the desktop window. Your PC does all the hard work (broken fonts, OCR, equations, voice). It writes one `.epub` holding every page of the PDF as it was printed, figures and equations included, plus the narration. The phone only plays it, so it works in airplane mode.

To get it onto the phone: put the `.epub` in iCloud Drive or OneDrive (or AirDrop / email it to yourself). On the iPhone, open the Files app, tap the file, tap Share, then tap Books. In Books, use the speaker (Read Aloud) button to start. The sentence being read is highlighted in yellow, and you can turn on automatic page turns in the Read Aloud settings.

| Choice | Why / evidence |
|---|---|
| EPUB 3 Media Overlays (SMIL) | The open standard for text synced to recorded audio. The reader adds the `media:active-class` class to the sentence being read ([W3C EPUB Media Overlays](https://w3c.github.io/epub-specs/archive/epub32/spec/epub-mediaoverlays.html)). |
| Fixed-layout pages | Apple Books plays Read Aloud "only in fixed-layout books" ([Apple Books Asset Guide](https://help.apple.com/itc/booksassetguide/en.lproj/itc6db639756.html)). It doesn't for reflowable EPUBs ([MobileRead test](https://www.mobileread.com/forums/showthread.php?p=4227408)). |
| Page image + highlight boxes | Shows the real page with nothing re-typeset. The boxes come from the line positions kept during extraction. Where a sentence starts mid-line, the position is estimated by character count, so the edge can be a few letters off. |
| Whole pages, nothing trimmed | The page is exactly as printed; pinch to zoom in Books. An iPhone 15 screen is 2.56 in wide (1179 px at 460 ppi), so a whole Letter page shows at about 30% of print size before zooming. `--crop` trims the white margins (about 40%). |
| Speed set at export (`--rate`) | Books' Read Aloud has no speed control, so the speed is built into the narration. Piper's `length_scale` stretches phoneme durations in the model itself (it is 1/rate), so faster speech keeps the same pitch, unlike speeding up a recording ([Piper voice config](https://github.com/rhasspy/piper/blob/master/TRAINING.md)). |
| `--strips 3` | Turned sideways, the screen is about 5.5 x 2.56 in, so a third of a page shows at about 85% of print size. Cuts snap to blank rows so no line is sliced. A sentence that crosses a cut is split, and its audio is divided by length. |
| AAC .m4a audio | A core EPUB audio type, and the format in Apple's Read Aloud samples. |
| Validated | Every sample EPUB passes [EPUBCheck](https://github.com/w3c/epubcheck) 5.1.0 with 0 errors and 0 warnings. `tests/test_epub.py` checks that every audio clip fits its file and every box fits its page. |

Known limits: pages from sideways scans that had to be rotated for OCR are shown but not highlighted. Size is about 21 MB per hour of narration plus about 150 KB per page.

## Install

1. **Python 3.10+**, then from this folder: `pip install -e ".[math]"`. The `[math]` part adds equation recognition, which pulls in PyTorch (about 200 MB on CPU). Leave it out if your books have no math.
2. **Tesseract OCR**, used for scanned pages and broken-font pages:
   - Windows: the installer from https://github.com/UB-Mannheim/tesseract/wiki, then add it to PATH
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`
   - Other languages: install the language pack and pass `--lang eng+fra`
3. **ffmpeg**, for MP3 and M4B export (`brew install ffmpeg`, `winget install ffmpeg`, or `apt install ffmpeg`).
4. **Node.js** (optional, recommended for math). It gives you natural equation speech. On first use the tool installs the Speech Rule Engine into `~/.pdfaudiobook/node`. Without Node, a simpler built-in reader handles equations.

On first run the Piper voice (`en_US-lessac-medium`, about 60 MB) and the math model (about 115 MB) download automatically. After that everything runs offline.

## How it works, and why each choice

| Step | What it does | Why (evidence) |
|---|---|---|
| 1. Text layer first | Reads the PDF's own text with PyMuPDF. | The text is exact when the font maps correctly, and it is about 100x faster than OCR. |
| 2. Garbled-text check | Scores each page from 0 to 1. The score counts broken characters (U+FFFD, private-use codes) and checks how many words are common English words. Pages below 0.45 go to OCR. | A PDF stores glyph IDs, and readers turn them into letters through the font's **ToUnicode CMap** (ISO 32000-1 §9.10). When that map is missing or wrong, you get the "weird font" junk. The top 100 English words make up about **50% of the Oxford English Corpus** ([source](https://en.wikipedia.org/wiki/Most_common_words_in_English)), so real prose scores high and shifted or garbled letters score near zero. |
| 3. OCR fallback | Renders the page at 300 DPI and runs Tesseract (LSTM engine). It keeps whichever version, text layer or OCR, scores higher. If OCR still looks garbled, it asks Tesseract which way the page is turned (`--psm 0`, orientation detection), rotates it, and tries again. | Tesseract: Smith, *An Overview of the Tesseract OCR Engine*, ICDAR 2007. Its docs say it "works best on images which have a DPI of at least 300 dpi" ([ImproveQuality](https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality)). Your *Confessions* scan is sideways two-page spreads: without rotation, OCR returned gibberish. |
| 3b. Symbols with broken or missing codes | When a glyph has no Unicode, it looks up the glyph's name inside the embedded font (`integraldisplay`, `parenleftbig`) and maps it with the Adobe Glyph List. When the name is meaningless (`H11001`), or a Greek font names its letters `a, b, c…`, it compares the glyph's outline with reference symbols and picks the closest one. It only renames a glyph when one candidate clearly wins. | Glyph names: [Adobe Glyph List](https://github.com/adobe-type-tools/agl-aglfn), read with fontTools. Outline matching is classical template matching (Duda & Hart, *Pattern Classification and Scene Analysis*, 1973). Tuned on Craig's Greek font: 20 of 26 letters correct, and the rest left unchanged rather than guessed. |
| 4. Cleanup | Removes running headers and page numbers (text that repeats in the top or bottom 8% of the page on 25% or more of pages). Rejoins wrapped lines, undoes end-of-line hyphenation, drops footnote markers, and joins rows that the PDF split into fragments. | Without this, the voice reads "THE QUIET TOWN, 47" on every page. |
| 5. Chapters | Uses (a) the PDF bookmarks if present; otherwise (b) headings like "Chapter 7" / "PART TWO", or lines set at 1.3x or more of body text size; otherwise (c) groups of 15 pages. With bookmarks, it goes one level deeper at a time until entries average 40 pages or fewer, so a book bookmarked as Parts → Chapters gives chapters. Cover, copyright, contents and index are listed but not read (`--all` reads them). | Bookmarks are the publisher's own structure (ISO 32000-1 §12.3.3). Ranking by font size is the same signal `pymupdf4llm`'s `IdentifyHeaders` uses. The 40-page cutoff is a usability rule of thumb (about 2 hours of audio), not a research result. |
| 6. Equations | **Typeset PDFs (most textbooks):** it rebuilds each equation as LaTeX from the PDF's own glyph positions. Smaller raised glyphs become exponents, smaller lowered glyphs become subscripts, a dot above a letter becomes ẋ, glyphs above and below a thin rule become a fraction, and limits above and below ∑ become ∑ from i=1 to n. This also covers math inside sentences ("solutions of the form x = e^(λt)"). **Matrices, big brackets, scans, and fonts that can't be decoded:** it crops the equation and recognizes it with **pix2tex**, running at several scales and keeping the consensus result. | TeX math fonts decode as junk (a minus sign comes out as "¡"). Radicals and fraction bars are drawn as lines, not text, and Tesseract reads text lines, not 2-D layouts. Rebuilding from glyph geometry follows Baker, Sexton & Sorge, *A Linear Grammar Approach to Mathematical Formula Recognition from PDF* (MKM 2009). It is exact when the glyphs decode, and about 100x faster than the image model: Goodwine went from more than 10 minutes for 12 pages to about 4 seconds per page. pix2tex is a ViT encoder + Transformer decoder; its README reports BLEU 0.88 and normalized edit distance 0.10 ([pix2tex](https://pypi.org/project/pix2tex)). Ink-blob segmentation is run-length smoothing (Wong, Casey & Wahl, *IBM J. Res. Dev.* 26(6), 1982). |
| 6b. Figures are not formulas | Labels inside diagrams ("A", "L*", "(b)", axis numbers) are not read, and a region with drawn curves or slanted/vertical strokes is never sent to the image model. Image-model output full of tokens that print never uses (`\stackrel{\cal D}{\P}`, `\star`, empty groups) is thrown away. | In a typeset formula the only drawn strokes are horizontal (fraction bars, radical overlines); brackets and operators are font glyphs. On 20 regions of Craig that the detector called math, the 7 real formulas had **0** curved, vertical or slanted strokes and 12 of the 13 diagrams had **12 or more**. Separating text from graphics is the standard first step of page segmentation (Fletcher & Kasturi, *IEEE PAMI* 10(6), 1988); here it is done on the PDF's vector paths. |
| 7. Math to speech | LaTeX goes through MathJax to MathML, then the **Speech Rule Engine** in ClearSpeak style: *"x equals negative b plus or minus the square root of b squared minus 4 a c, all over 2 a"*. | SRE (Sorge et al., *Towards Making Mathematics a First Class Citizen in General Screen Readers*, W4A 2014) is the math speech used in MathJax and Google ChromeVox ([Univ. of Birmingham](https://www.birmingham.ac.uk/research/centres-institutes/research-in-computer-science/accessibility-to-scientific-documents-for-visually-impaired-readers)). |
| 8. Voice | **Piper** neural TTS, offline. Falls back to your OS voice, then eSpeak NG. | Piper uses VITS (Kim, Kong & Son, ICML 2021, [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)), which "achieves a MOS comparable to ground truth" in listening tests. Text goes in groups of a few sentences (400 characters or fewer), because these models are trained on sentence-length clips. |
| 9. Desktop window | Tkinter window; passages are synthesized one ahead on a background thread while the current one plays; the passage being spoken is highlighted. | Tkinter ships with Python on Windows, macOS and Linux, so the window adds no install step. Highlighting the words being spoken is how synchronized text-and-audio books work in the DAISY standard and EPUB 3 Media Overlays (W3C). Making the next passage during playback hides the synthesis time, so there is no gap between passages. |
| 10. Output | One MP3 per chapter (64 kbps mono) plus a playlist, and one **.m4b** (AAC, 48 kbps) with a chapter table. | M4B chapter markers show up as a chapter list in Apple Books, BookPlayer, Smart AudioBook Player and VLC. The markers are written with ffmpeg's FFMETADATA format ([docs](https://ffmpeg.org/ffmpeg-formats.html#Metadata-2)). |

## Measured on your books (October 2026)

These are 10–12 page excerpts from PDFs in your Downloads, each checked by eye against the rendered page:

| Book | What was wrong | Now |
|---|---|---|
| Goodwine, *Engineering Differential Equations* (Springer, LaTeX) | Exponents flattened ("e−αt"); TeX big brackets decoded as "�"; each equation took ~1 min on the image model | All 24 display equations on the two pages checked (pp. 7 and 11 of the excerpt) came out right, for example *x sub p of t equals negative 2 over 13 cosine 2 t plus 3 over 13 sine 2 t*. About 4 s/page; matrices still go to the image model. |
| Borgnakke & Sonntag, *Fundamentals of Thermodynamics* (Wiley, STIX fonts) | Dotted variables (ṁ) and equation numbers mangled | The 8 equations on the page checked are right, with "(Equation 3.7)" read after each; 1.7 s/page |
| Craig, *Mechanics of Materials* (Wiley, Mathematical Pi fonts) | =, −, σ, ∑, ∫ came out as "�" or wrong letters | σ, =, −, +, ×, ∫, ∑, Δ, big brackets, → and … are decoded. Diagram labels are no longer read as equations: on pp. 40–70, 165 "equation" lines dropped to 47 real ones, image-model nonsense (e.g. "Λ-bar prime mu phi") went to zero, and the time went from 417 s to 122 s (4 s/page). The Δ that came out as Λ was a bug in the reference list (the Adobe Glyph List gives "Delta" the code for the increment sign, so Δ was never a candidate). |
| *Confessions*, Book 1 (sideways scan, two pages per image) | OCR returned gibberish; 561 s | Clean text, 85 s for 10 images, and all 18 chapters plus "Book One" found |
| Wilson, *Sociobiology* (prose) | "Chapters" were Parts lasting 5–9 hours | 35 chapter-sized entries chosen automatically |

A whole textbook is a one-time job of roughly 2–8 seconds per page. The result is cached, so later reads start immediately. Use `--fast-math` to skip the image model and have complex layouts announced as "an equation appears here".

## Measured on the test books

`tests/make_samples.py` builds books that reproduce each failure. `python -m pytest tests` checks them (19 tests, all passing):

| Sample | Problem | Result |
|---|---|---|
| clean.pdf | normal PDF with bookmarks | 3/3 chapters from bookmarks; headers and page numbers removed |
| garbled.pdf | embedded font with a corrupted ToUnicode map (copy-paste gives `Aol Xbpla Avdu`) | garbled text detected; 7/7 pages OCR'd; 3/3 chapters found from headings; word accuracy above 97% (1 - WER) |
| scanned.pdf | image-only pages | 3/3 chapters; word accuracy above 97%; no false equations in plain prose |
| math.pdf / math_scanned.pdf | Computer Modern equations, typeset and scanned | 3/3 equations found and 3/3 chapters. `E = mc²` and Σi = n(n+1)/2 recognized exactly. The quadratic formula came out as x = −(b ± √…)/2a on the scan, which has the same values in a different form, and with a sign error in the typeset copy. |

**Known limits, stated honestly:**
- The image model (pix2tex) is the weakest link. On matrices and long formulas it sometimes drops a sign or garbles a row, so check `text --spoken` for a math-heavy chapter.
- Craig's italic π (in its Greek font, coded as "p") does not match the upright reference π clearly enough, so it is left as "p" rather than guessed.
- When a letter is also used as a function in the same formula, x(t) = x₀e^(…), the product is read with an explicit "times" ("x sub 0 times e to the…"), because SRE otherwise reads it as "x sub 0 of e".
- Free-body-diagram equations with sign-convention arrows (+→ ΣF = 0) still go to the image model and come out unevenly.
- Pages with complex multi-column layouts can come out in the wrong reading order.

## Useful options

- `--ocr text|ocr|auto`: never OCR, always OCR, or decide per page (default).
- `--chapters outline|headings|pages` and `--level 2`: force a strategy, or use second-level bookmarks as chapters.
- `--engine piper|system|espeak`, `--voice en_US-ryan-high`, `--rate 1.25`: change the voice and speed. Voices are listed at https://huggingface.co/rhasspy/piper-voices.
- `--no-math`: skip equation detection, which is faster for novels. `--fast-math`: keep equations but skip the slow image model.
- `--all`: also read cover, copyright, contents and index.
- Results are cached in `~/.pdfaudiobook/cache`, so OCR runs once per book.
