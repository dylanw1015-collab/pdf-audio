"""Build three sample books that reproduce the failure cases:

clean.pdf    - normal text layer, PDF bookmarks, running headers and page numbers
garbled.pdf  - embedded subset font with a corrupted ToUnicode map, so copying
               text gives junk although the page looks fine; no bookmarks
scanned.pdf  - pages are images only (no text layer at all)
math.pdf     - prose with display equations set in Computer Modern math fonts
"""

import os
import re
import sys

import pymupdf
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PARA = (
    "It was a bright cold day in the spring, and the town was quiet as the people went about "
    "their work. She had never thought that the old house at the end of the road would be the "
    "place where her life would change. There was something in the way the light came through "
    "the windows that made her stop and look again. "
)
CHAPTERS = [("Chapter One", "The Old House"), ("Chapter Two", "A Letter Arrives"), ("Chapter Three", "The Road North")]


def write_book(path, font="Times-Roman", bold="Times-Bold", outline=True):
    c = canvas.Canvas(path, pagesize=letter)
    W, H = letter
    page = 1

    def furniture():
        c.setFont(font, 9)
        c.drawString(72, H - 40, "THE QUIET TOWN")  # running header
        c.drawCentredString(W / 2, 36, str(page))  # page number

    c.setFont(bold, 28)
    c.drawCentredString(W / 2, H / 2, "The Quiet Town")
    c.showPage()
    for n, (head, sub) in enumerate(CHAPTERS):
        for k in range(2):  # two pages per chapter
            page += 1
            furniture()
            y = H - 110
            if k == 0:
                if outline:
                    c.bookmarkPage(f"ch{n}")
                    c.addOutlineEntry(f"{head}: {sub}", f"ch{n}", level=0)
                c.setFont(bold, 22)
                c.drawString(72, y, head)
                c.setFont(bold, 16)
                c.drawString(72, y - 28, sub)
                y -= 80
            c.setFont(font, 11)
            text = (PARA * 3).split()
            for para in range(3):
                line = ""
                for w in text:
                    if c.stringWidth(line + " " + w, font, 11) > W - 144:
                        c.drawString(72, y, line.strip())
                        y -= 15
                        line = ""
                    line += " " + w
                c.drawString(72, y, line.strip())
                y -= 30
                if y < 120:
                    break
            c.showPage()
    c.save()


def _shift(m):
    cp = int(m.group(2), 16)
    ch = chr(cp)
    if ch.isascii() and ch.isalpha():
        base = ord("a") if ch.islower() else ord("A")
        cp = base + (cp - base + 7) % 26
    return f"{m.group(1)} <{cp:04X}>"


def break_tounicode(src, dst):
    """Rewrite each font's ToUnicode map so letters decode shifted by 7
    (e becomes l). The page renders correctly but extracted text is junk,
    the same symptom as a publisher's broken CMap."""
    r = PdfReader(src)
    w = PdfWriter()
    for p in r.pages:
        fonts = p["/Resources"].get("/Font", {})
        for f in fonts.values():
            f = f.get_object()
            if "/ToUnicode" in f:
                stream = f["/ToUnicode"].get_object()
                data = stream.get_data().decode("latin-1")
                data = re.sub(r"(<[0-9A-Fa-f]{2}>)\s*<([0-9A-Fa-f]{4})>", _shift, data)
                stream.set_data(data.encode("latin-1"))
        w.add_page(p)
    with open(dst, "wb") as fh:
        w.write(fh)


def rasterize(src, dst, dpi=200):
    s = pymupdf.open(src)
    d = pymupdf.open()
    for p in s:
        pix = p.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        np = d.new_page(width=p.rect.width, height=p.rect.height)
        np.insert_image(np.rect, pixmap=pix)
    d.save(dst)


EQUATIONS = [
    r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
    r"E = mc^2",
    r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
]


def write_math_book(path):
    """One page per equation, typeset with matplotlib's mathtext (a TeX
    subset) in Computer Modern, the same fonts LaTeX papers embed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42
    with PdfPages(path) as pdf:
        for n, eq in enumerate(EQUATIONS, 1):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.9, f"Chapter {n}", fontsize=22, weight="bold")
            fig.text(0.1, 0.8, "The result we need is the following equation.", fontsize=12)
            fig.text(0.5, 0.7, f"${eq}$", fontsize=18, ha="center")
            fig.text(0.1, 0.6, "We will use this result again in the next chapter.", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "samples"
    os.makedirs(out, exist_ok=True)
    write_book(f"{out}/clean.pdf")
    pdfmetrics.registerFont(TTFont("DVS", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
    pdfmetrics.registerFont(TTFont("DVSB", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"))
    write_book(f"{out}/_tmp.pdf", "DVS", "DVSB", outline=False)
    break_tounicode(f"{out}/_tmp.pdf", f"{out}/garbled.pdf")
    os.remove(f"{out}/_tmp.pdf")
    rasterize(f"{out}/clean.pdf", f"{out}/scanned.pdf")
    write_math_book(f"{out}/math.pdf")
    rasterize(f"{out}/math.pdf", f"{out}/math_scanned.pdf", dpi=300)
    print("ok")
