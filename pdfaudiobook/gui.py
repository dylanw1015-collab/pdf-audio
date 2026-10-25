"""Desktop reader window: open a PDF, pick a chapter, press play.

Built on Tkinter because it ships with Python on Windows, macOS and Linux,
so the window needs nothing beyond what the command-line tool already
installs. The text pane shows exactly the words being spoken (equations in
their spoken form) and highlights the passage being read, the way
synchronized text and audio work in DAISY talking books and EPUB 3 Media
Overlays (W3C, *EPUB 3 Media Overlays*, 2023).

Speech is made one passage (a few sentences, see tts.chunk_text) at a time
on a background thread, and the next passage is synthesized while the
current one plays, so there is no gap and Pause/Skip respond within one
passage. Your place in each book is saved in the same file the `read`
command uses, so the window and the command line resume from the same spot.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import wave
from tkinter import filedialog, messagebox, ttk

PROGRESS_FILE = os.path.join(os.path.expanduser("~"), ".pdfaudiobook", "progress.json")


class Player:
    """Plays one WAV at a time without blocking, and can be stopped."""

    def __init__(self):
        self.proc = None
        self.stop_evt = threading.Event()

    def play(self, path: str) -> bool:
        """Play to the end (True) or until stop() (False). Call from a worker thread."""
        self.stop_evt.clear()
        with wave.open(path) as w:
            secs = w.getnframes() / w.getframerate()
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            done = not self.stop_evt.wait(secs + 0.05)
            if not done:
                winsound.PlaySound(None, 0)
            return done
        for cmd in (["afplay"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
            if shutil.which(cmd[0]):
                self.proc = subprocess.Popen(cmd + [path])
                break
        else:
            raise RuntimeError("No audio player found (install ffmpeg for ffplay).")
        while self.proc.poll() is None:
            if self.stop_evt.wait(0.05):
                self.proc.terminate()
                return False
        return True

    def stop(self):
        self.stop_evt.set()


class Reader(tk.Tk):
    def __init__(self, pdf: str | None = None):
        super().__init__()
        self.title("PDF Audiobook")
        self.geometry("980x640")
        self.minsize(640, 420)
        self.book = None
        self.pdf = None
        self.ch = 0  # chapter index
        self.k = 0  # passage index within the chapter
        self.passages: list[str] = []
        self.prects: list[list] = []  # per passage: [(page, (x0, y0, x1, y1)), ...] in PDF points
        self.doc = None
        self.page_img = None  # (page, zoom, PhotoImage), kept so Tk does not drop it
        self.cur_page, self.cur_rects = 0, []
        self.playing = False
        self.player = Player()
        self.events: queue.Queue = queue.Queue()
        self.jobs: queue.Queue = queue.Queue()
        self.rate = tk.DoubleVar(value=1.0)
        self.status = tk.StringVar(value="Open a PDF to start.")
        self._build()
        threading.Thread(target=self._speech_worker, daemon=True).start()
        self.after(50, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        if pdf:
            self.after(100, lambda: self.open_pdf(pdf))

    # ------------------------------------------------------------ layout
    def _build(self):
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Open PDF…", command=self.open_pdf).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.btn_prev = ttk.Button(bar, text="⏮ Back", command=lambda: self.skip(-1))
        self.btn_play = ttk.Button(bar, text="▶ Play", command=self.toggle, width=10)
        self.btn_next = ttk.Button(bar, text="Next ⏭", command=lambda: self.skip(1))
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.pack(side="left", padx=2)
        ttk.Label(bar, text="Speed").pack(side="left", padx=(16, 4))
        ttk.Scale(bar, from_=0.6, to=2.0, variable=self.rate, length=140,
                  command=lambda _v: self.speed_lbl.config(text=f"{self.rate.get():.1f}×")).pack(side="left")
        self.speed_lbl = ttk.Label(bar, text="1.0×", width=5)
        self.speed_lbl.pack(side="left")
        ttk.Button(bar, text="Export audiobook…", command=self.export).pack(side="right")

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6)
        left = ttk.Frame(body)
        ttk.Label(left, text="Chapters").pack(anchor="w")
        self.chlist = tk.Listbox(left, activestyle="dotbox", exportselection=False, width=34)
        self.chlist.pack(fill="both", expand=True)
        self.chlist.bind("<<ListboxSelect>>", self._chapter_clicked)
        body.add(left, weight=1)
        tabs = ttk.Notebook(body)
        pagef = ttk.Frame(tabs)
        self.canvas = tk.Canvas(pagef, background="#555555", highlightthickness=0)
        csb = ttk.Scrollbar(pagef, command=self.canvas.yview)
        self.canvas.config(yscrollcommand=csb.set)
        csb.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._page_clicked)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self.canvas.bind("<Configure>", lambda _e: self.after(150, self._rerender))
        tabs.add(pagef, text="Page")
        right = ttk.Frame(tabs)
        tabs.add(right, text="Text")
        self.text = tk.Text(right, wrap="word", font=("Georgia", 13), padx=14, pady=10,
                            spacing2=3, spacing3=10, cursor="arrow")
        sb = ttk.Scrollbar(right, command=self.text.yview)
        self.text.config(yscrollcommand=sb.set, state="disabled")
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self.text.tag_config("now", background="#fff3b0")
        self.text.tag_bind("passage", "<Button-1>", self._passage_clicked)
        body.add(tabs, weight=3)

        foot = ttk.Frame(self, padding=(6, 4))
        foot.pack(fill="x")
        self.prog = ttk.Progressbar(foot, length=180, mode="determinate")
        self.prog.pack(side="right")
        ttk.Label(foot, textvariable=self.status).pack(side="left")
        self.bind("<space>", lambda e: self.toggle())
        self.bind("<Left>", lambda e: self.skip(-1))
        self.bind("<Right>", lambda e: self.skip(1))

    # ------------------------------------------------------------ book
    def open_pdf(self, path: str | None = None):
        path = path or filedialog.askopenfilename(filetypes=[("PDF books", "*.pdf")])
        if not path:
            return
        self.stop()
        self.status.set(f"Reading {os.path.basename(path)} (scanned pages take longer)…")

        def work():
            from .book import load_book

            try:
                b = load_book(path, progress=lambda i, n, _p: self.events.put(("progress", i, n)))
                self.events.put(("loaded", path, b))
            except Exception as e:  # show any failure in the window
                self.events.put(("error", f"Could not read this PDF:\n{e}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_book(self, path, b):
        from .chapters import is_skippable

        self.pdf, self.book = path, b
        if self.doc:
            self.doc.close()
        import pymupdf

        self.doc = pymupdf.open(path)
        self.title(f"{b['title']} · PDF Audiobook")
        self.chlist.delete(0, "end")
        for i, c in enumerate(b["chapters"]):
            self.chlist.insert("end", f"{i + 1}. {c.title}")
            if is_skippable(c):
                self.chlist.itemconfig(i, foreground="#999999")
        ch, k = self._load_place()
        if ch is None:  # start at the first chapter worth reading
            ch = next((i for i, c in enumerate(b["chapters"]) if not is_skippable(c)), 0)
            k = 0
        ocr = len(b.get("ocr_pages", []))
        self.status.set(f"{len(b['chapters'])} chapters found from {b['strategy']}"
                        + (f"; {ocr} pages read by OCR" if ocr else "") + ".")
        self.select_chapter(ch, k)

    def select_chapter(self, ch: int, k: int = 0):
        from .chapters import spoken_text

        self.ch = max(0, min(ch, len(self.book["chapters"]) - 1))
        self.chlist.selection_clear(0, "end")
        self.chlist.selection_set(self.ch)
        self.chlist.see(self.ch)
        self.passages = []
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", "Preparing the spoken text…")
        self.text.config(state="disabled")
        # Equations are turned into words by the speech thread (SRE lives there).
        self.jobs.put(("prepare", self.ch, spoken_text(self.book["chapters"][self.ch]), k))
        self.show_page(self.book["chapters"][self.ch].start_page, [])

    def _passage_rects(self, passages, owners, nparts):
        """Page rectangles for each passage, by its share of its paragraph's characters.

        Equation rewriting changes text length, so this is approximate (it picks
        whole lines). A passage gets [] when its layout is unknown.
        """
        boxes = self.book["chapters"][self.ch].boxes
        off = nparts - len(boxes)  # 1 when the spoken title was added in front
        out = []
        for j, p in enumerate(passages):
            pi = owners[j] - off
            if off not in (0, 1) or not 0 <= pi < len(boxes):
                out.append([])
                continue
            mine = [i for i, o in enumerate(owners) if o == owners[j]]
            total = sum(len(passages[i]) for i in mine) or 1
            f0 = sum(len(passages[i]) for i in mine if i < j) / total
            f1 = f0 + len(p) / total
            lines, n = boxes[pi], sum(b[5] for b in boxes[pi]) or 1
            cum, rects = 0, []
            for q, x0, y0, x1, y1, nc in lines:
                if cum < f1 * n and cum + nc > f0 * n and x1 > x0:
                    rects.append((q, (x0, y0, x1, y1)))
                cum += nc
            out.append(rects)
        return out

    def _show_passages(self, ch, passages, k, owners=(), nparts=0):
        if ch != self.ch:
            return
        self.passages = passages
        self.prects = self._passage_rects(passages, owners, nparts) if owners else [[] for _ in passages]
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        for i, p in enumerate(passages):
            self.text.insert("end", p, ("passage", f"p{i}"))
            self.text.insert("end", "\n\n")
        self.text.config(state="disabled")
        self.highlight(min(k, max(len(passages) - 1, 0)))
        if self.playing:
            self._speak_current()

    def highlight(self, k: int):
        self.k = k
        self.text.tag_remove("now", "1.0", "end")
        r = self.text.tag_ranges(f"p{k}")
        if r:
            self.text.tag_add("now", r[0], r[1])
            self.text.see(r[0])
        rects = self.prects[k] if k < len(self.prects) else []
        if rects:
            self.show_page(rects[0][0], [r for q, r in rects if q == rects[0][0]])
        self._save_place()

    # ------------------------------------------------------------ page view
    def show_page(self, pno, rects):
        self.cur_page, self.cur_rects = pno, rects
        self._rerender()
        if rects and self.page_img:
            self.canvas.yview_moveto(max(0, (rects[0][1] * self.page_img[1] - 60) / self.page_img[2].height()))

    def _rerender(self):
        if not self.doc:
            return
        import pymupdf

        pno = max(0, min(self.cur_page, len(self.doc) - 1))
        page = self.doc[pno]
        z = max(self.canvas.winfo_width(), 300) / page.rect.width
        if not self.page_img or self.page_img[:2] != (pno, z):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
            self.page_img = (pno, z, tk.PhotoImage(data=pix.tobytes("ppm")))
        self.canvas.delete("all")
        img = self.page_img[2]
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        for x0, y0, x1, y1 in self.cur_rects:
            self.canvas.create_rectangle(x0 * z, y0 * z, x1 * z, y1 * z, outline="#e08000", width=2,
                                         fill="#ffd400", stipple="gray25")
        self.canvas.config(scrollregion=(0, 0, img.width(), img.height()))

    def _page_clicked(self, e):
        if not self.page_img:
            return
        pno, z, _ = self.page_img
        x, y = self.canvas.canvasx(e.x) / z, self.canvas.canvasy(e.y) / z
        best, dist = None, 1e9
        for k, rects in enumerate(self.prects):
            for q, (x0, y0, x1, y1) in rects:
                if q == pno:
                    d = max(x0 - x, 0, x - x1) + max(y0 - y, 0, y - y1)
                    if d < dist:
                        best, dist = k, d
        if best is not None:
            self.player.stop()
            self.highlight(best)
            if self.playing:
                self._speak_current()

    # ------------------------------------------------------------ playback
    def toggle(self):
        if not self.book:
            return
        if self.playing:
            self.stop()
        else:
            self.playing = True
            self.btn_play.config(text="⏸ Pause")
            self._speak_current()

    def stop(self):
        self.playing = False
        self.btn_play.config(text="▶ Play")
        self.player.stop()

    def skip(self, d: int):
        if not self.passages:
            return
        k = self.k + d
        if k < 0 and self.ch > 0:
            return self.select_chapter(self.ch - 1, 10**9)
        if k >= len(self.passages):
            return self._next_chapter()
        self.player.stop()
        self.highlight(max(k, 0))
        if self.playing:
            self._speak_current()

    def _next_chapter(self):
        from .chapters import is_skippable

        nxt = next((i for i in range(self.ch + 1, len(self.book["chapters"]))
                    if not is_skippable(self.book["chapters"][i])), None)
        if nxt is None:
            self.stop()
            self.status.set("Finished the book.")
            return
        self.player.stop()
        self.select_chapter(nxt, 0)

    def _speak_current(self):
        if self.passages:
            self.jobs.put(("speak", self.ch, self.k, self.passages, round(self.rate.get(), 1)))

    def _speech_worker(self):
        """Owns the TTS engine and the math speaker; runs every speech job."""
        if sys.platform == "win32":
            try:  # SAPI voices are COM objects and need COM on this thread
                import pythoncom

                pythoncom.CoInitialize()
            except ImportError:
                pass
        from .math import MathSpeaker
        from .tts import chunk_text, get_engine

        speaker, engine, eng_rate = None, None, None
        tmp = tempfile.mkdtemp(prefix="pdfab_gui_")
        made: dict[tuple, str] = {}
        try:  # load the voice while the book loads, so the first Play is not silent for ~10 s
            self.events.put(("status", "Loading the voice…"))
            engine, eng_rate = get_engine(rate=1.0), 1.0
            self.events.put(("status", f"Voice: {engine.name}"))
        except Exception:
            engine = None  # the first Play retries and shows the error
        while True:
            job = self.jobs.get()
            # Only the newest speak job matters; drop ones superseded by Skip.
            while job[0] == "speak" and not self.jobs.empty():
                job = self.jobs.get()
            try:
                if job[0] == "prepare":
                    _, ch, text, k = job
                    speaker = speaker or MathSpeaker()
                    # Same passages as the read command (chunk_text splits on blank
                    # lines first), kept per paragraph so each maps back to the page.
                    parts = text.split("\n\n")
                    passages, owners = [], []
                    for i, part in enumerate(parts):
                        c = chunk_text(speaker.speak_text(part))
                        passages += c
                        owners += [i] * len(c)
                    self.events.put(("passages", ch, passages, k, owners, len(parts)))
                    continue
                _, ch, k, passages, rate = job
                if engine is None or rate != eng_rate:
                    self.events.put(("status", "Loading the voice…"))
                    engine, eng_rate, made = get_engine(rate=rate), rate, {}
                    self.events.put(("status", f"Voice: {engine.name}, speed {rate}×"))

                def wav_for(i):
                    key = (ch, i)
                    if key not in made:
                        p = os.path.join(tmp, f"{ch}_{i}.wav")
                        engine.to_wav(passages[i], p)
                        made[key] = p
                    return made[key]

                wav = wav_for(k)
                if k + 1 < len(passages):  # make the next passage while this one plays
                    threading.Thread(target=wav_for, args=(k + 1,), daemon=True).start()
                if not self.playing or ch != self.ch or k != self.k:
                    continue
                if self.player.play(wav) and self.playing:
                    self.events.put(("advance", ch, k))
            except Exception as e:
                self.events.put(("error", f"Speech failed: {e}"))

    def _pump(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "progress":
                    self.prog.config(maximum=ev[2], value=ev[1])
                elif kind == "loaded":
                    self.prog.config(value=0)
                    self._show_book(ev[1], ev[2])
                elif kind == "passages":
                    self._show_passages(*ev[1:])
                elif kind == "advance" and ev[1] == self.ch and ev[2] == self.k:
                    self.skip(1)
                elif kind == "status":
                    self.status.set(ev[1])
                elif kind == "error":
                    self.stop()
                    messagebox.showerror("PDF Audiobook", ev[1])
        except queue.Empty:
            pass
        self.after(50, self._pump)

    # ------------------------------------------------------------ events
    def _chapter_clicked(self, _e):
        sel = self.chlist.curselection()
        if sel and sel[0] != self.ch and self.book:
            self.player.stop()
            self.select_chapter(sel[0], 0)

    def _passage_clicked(self, e):
        for t in self.text.tag_names(f"@{e.x},{e.y}"):
            if t.startswith("p") and t[1:].isdigit():
                self.player.stop()
                self.highlight(int(t[1:]))
                if self.playing:
                    self._speak_current()

    def export(self):
        if not self.book:
            return
        out = filedialog.askdirectory(title="Folder for the audiobook files")
        if not out:
            return
        self.stop()
        self.status.set("Exporting: one MP3 per chapter plus an M4B with chapter marks…")

        def work():
            from .chapters import is_skippable
            from .export import export_book
            from .tts import get_engine

            keep = [i + 1 for i, c in enumerate(self.book["chapters"]) if not is_skippable(c)]
            try:
                files = export_book(self.book, get_engine(rate=round(self.rate.get(), 1)), out, fmt="mp3+m4b",
                                    chapters=keep, log=lambda m: self.events.put(("status", str(m))))
                self.events.put(("status", f"Exported {len(files)} files to {out}"))
            except Exception as e:
                self.events.put(("error", f"Export failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ place
    def _load_place(self):
        try:
            with open(PROGRESS_FILE) as f:
                ch, k = json.load(f)[os.path.abspath(self.pdf)]
            return ch - 1, k  # stored 1-based, like the read command
        except (OSError, ValueError, KeyError, TypeError):
            return None, 0

    def _save_place(self):
        if not self.pdf:
            return
        try:
            with open(PROGRESS_FILE) as f:
                prog = json.load(f)
        except (OSError, ValueError):
            prog = {}
        prog[os.path.abspath(self.pdf)] = (self.ch + 1, self.k)
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(prog, f)

    def _quit(self):
        self.player.stop()
        self.destroy()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    Reader(argv[0] if argv else None).mainloop()


if __name__ == "__main__":
    main()
