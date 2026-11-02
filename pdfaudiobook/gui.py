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

    def _play_stream(self, path: str) -> bool:
        """Write the passage into one long-lived output stream. Opening a device per passage costs
        over a second on some Windows drivers (a pause) and a timed stop clips the last words."""
        import numpy as np

        with wave.open(path) as w:
            sr, ch = w.getframerate(), w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
        if getattr(self, "_fmt", None) != (sr, ch):
            if getattr(self, "_out", None):
                self._out.close()
            self._out = self._sd.OutputStream(samplerate=sr, channels=ch, dtype="int16")
            self._out.start()
            self._fmt = (sr, ch)
        for i in range(0, len(data), sr // 10):
            if self.stop_evt.is_set():
                self._out.abort()  # drops what is buffered, so Pause/Skip are instant
                self._out.start()
                return False
            self._out.write(data[i:i + sr // 10])
        return True

    def play(self, path: str) -> bool:
        """Play to the end (True) or until stop() (False). Call from a worker thread."""
        self.stop_evt.clear()
        try:
            import sounddevice as _sd

            self._sd = _sd
        except (ImportError, OSError):
            _sd = None
        if _sd:
            return self._play_stream(path)
        with wave.open(path) as w:
            secs = w.getnframes() / w.getframerate()
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            done = not self.stop_evt.wait(secs + 0.5)  # extra for the driver's start-up delay
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
        self.page_range = (0, 0)  # first and last page of the chapter shown in the Page tab
        self.layout = None  # (key, total height, {page: (x, y, height, zoom)}): where each page sits on the canvas
        self.pimgs: dict[int, tk.PhotoImage] = {}  # pages drawn right now; kept so Tk does not drop them
        self.cur_page, self.cur_rects = 0, []
        self.playing = False
        self.gen = 0  # bumped on every (re)start of speech; the worker drops anything older
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
        ttk.Button(bar, text="Export for iPhone…", command=self.export_epub).pack(side="right")
        ttk.Button(bar, text="⛶ Full screen (F11)", command=self.toggle_fullscreen).pack(side="right", padx=4)

        body = self.body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6)
        left = self.left = ttk.Frame(body)
        ttk.Label(left, text="Chapters").pack(anchor="w")
        self.chlist = tk.Listbox(left, activestyle="dotbox", exportselection=False, width=34)
        self.chlist.pack(fill="both", expand=True)
        self.chlist.bind("<<ListboxSelect>>", self._chapter_clicked)
        body.add(left, weight=1)
        tabs = self.tabs = ttk.Notebook(body)
        pagef = ttk.Frame(tabs)
        self.canvas = tk.Canvas(pagef, background="#555555", highlightthickness=0, yscrollincrement=40)
        csb = ttk.Scrollbar(pagef, command=self.canvas.yview)
        self.canvas.config(yscrollcommand=lambda a, b: (csb.set(a, b), self.after_idle(self._rerender)))
        csb.pack(side="right", fill="y")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._page_clicked)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-3 * (e.delta // 120), "units"))
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
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.toggle_fullscreen(False))
        # Space pauses from anywhere. Widgets have their own space action (a focused button
        # presses itself, the text pane eats the key), which made it fire twice or not at all.
        pause = lambda e: (self.toggle(), "break")[1]
        self.bind("<space>", pause)
        for cls in ("TButton", "Listbox", "Text", "Canvas", "TNotebook", "TScale"):
            self.bind_class(cls, "<space>", pause)
        self.bind_class("TButton", "<KeyRelease-space>", lambda e: "break")
        self.bind("<Left>", lambda e: self.skip(-1))
        self.bind("<Right>", lambda e: self.skip(1))

    def toggle_fullscreen(self, on=None):
        """Full screen gives the page the whole window: the chapter list is hidden until Esc."""
        on = not self.attributes("-fullscreen") if on is None else on
        if on == bool(self.attributes("-fullscreen")):
            return
        self.attributes("-fullscreen", on)
        if on:
            self.body.forget(self.left)
            self.tabs.select(0)
        else:
            self.body.insert(0, self.left, weight=1)

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
                # shortcut: skip the slow image model (pix2tex, ~25 s per hard equation); most equations are
                # rebuilt from glyph positions in milliseconds, hard ones (matrices) become "an equation appears
                # here". Upgrade: re-read those with pix2tex in the background per chapter.
                from .math import MathOCR

                MathOCR.disabled = True
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
        c = self.book["chapters"][self.ch]
        self.page_range = (c.start_page, c.end_page)
        self.show_page(c.start_page, [])

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
        """Show the page being read. The view only moves when the passage is off screen, and then it
        snaps to the top of that page's spread, so the text stays still while you read it."""
        self.cur_page, self.cur_rects = pno, rects
        lo, hi = self.page_range
        if not lo <= pno <= hi:  # a passage on a page outside the shown range: widen it
            self.page_range = (min(lo, pno), max(hi, pno))
        self._rerender()
        _x, y, h, z = self.layout[2][pno]
        top, view = self.canvas.canvasy(0), max(self.canvas.winfo_height(), 1)
        r0 = y + (rects[0][1] * z if rects else 0)
        r1 = y + (rects[-1][3] * z if rects else 0)
        if not rects or r0 < top or r1 > top + view:
            self.canvas.yview_moveto(y / self.layout[1])
            if rects and r1 > y + view:  # the page is taller than the window: bring the passage into view
                self.canvas.yview_moveto(max(0, r0 - 40) / self.layout[1])
        self._rerender()

    def _rerender(self, force=False):
        """Lay the chapter out as book spreads (two pages across, left page = odd index,
        so page 1 stands alone on the right like a real book); draw only spreads in view."""
        if not self.doc:
            return
        import pymupdf

        lo, hi = self.page_range
        hi = min(hi, len(self.doc) - 1)
        w = max(self.canvas.winfo_width(), 400)
        key = (lo, hi, w)
        if force or not self.layout or self.layout[0] != key:
            half = w / 2
            geo, y = {}, 0
            for a in range(lo if lo % 2 else lo - 1, hi + 1, 2):  # a = left page of the spread
                row = [n for n in (a, a + 1) if lo <= n <= hi]
                zs = {n: half / self.doc[n].rect.width for n in row}
                rowh = max(self.doc[n].rect.height * zs[n] for n in row)
                for n in row:
                    geo[n] = (0 if n == a else half, y, self.doc[n].rect.height * zs[n], zs[n])
                y += rowh + 10
            self.layout = (key, y, geo)
            self.pimgs.clear()
            self.canvas.delete("all")
            self.canvas.config(scrollregion=(0, 0, w, y))
        geo = self.layout[2]
        top, bot = self.canvas.canvasy(0), self.canvas.canvasy(self.canvas.winfo_height())
        pad = bot - top  # one screen of spreads ahead and behind, so scrolling shows no gaps
        for n, (x, y, h, z) in geo.items():
            seen = y + h >= top - pad and y <= bot + pad
            if seen and n not in self.pimgs:
                pix = self.doc[n].get_pixmap(matrix=pymupdf.Matrix(z, z))
                self.pimgs[n] = tk.PhotoImage(data=pix.tobytes("ppm"))
                self.canvas.create_image(x, y, anchor="nw", image=self.pimgs[n], tags=f"pg{n}")
            elif not seen and n in self.pimgs:
                self.canvas.delete(f"pg{n}")
                del self.pimgs[n]
        self.canvas.delete("hl")
        if self.cur_page in geo:
            x, y, _h, z = geo[self.cur_page]
            for x0, y0, x1, y1 in self.cur_rects:
                self.canvas.create_rectangle(x + x0 * z, y + y0 * z, x + x1 * z, y + y1 * z, outline="#e08000",
                                             width=2, fill="#ffd400", stipple="gray25", tags="hl")

    def _page_clicked(self, e):
        if not self.layout:
            return
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        half = self.layout[0][2] / 2
        hit = next((n for n, (x, y, h, _z) in self.layout[2].items() if x <= cx < x + half and y <= cy < y + h), None)
        if hit is None:
            return
        x, y, _h, z = self.layout[2][hit]
        px, py = (cx - x) / z, (cy - y) / z
        best, dist = None, 1e9
        for k, rects in enumerate(self.prects):
            for q, (x0, y0, x1, y1) in rects:
                if q == hit:
                    d = max(x0 - px, 0, px - x1) + max(y0 - py, 0, py - y1)
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
            self.gen += 1
            self.jobs.put(("speak", self.ch, self.k, self.passages, round(self.rate.get(), 1), self.gen))

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
        made: dict[str, str] = {}
        synth_lock = threading.Lock()
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
                _, ch, k, passages, rate, gen = job
                if engine is None or rate != eng_rate:
                    self.events.put(("status", "Loading the voice…"))
                    engine, eng_rate, made = get_engine(rate=rate), rate, {}
                    self.events.put(("status", f"Voice: {engine.name}, speed {rate}×"))

                def wav_for(i):
                    key = passages[i]  # by text, so a stale passage number can never play the wrong audio
                    with synth_lock:  # the voice is not thread-safe; the prefetch thread must not overlap
                        if key not in made:
                            p = os.path.join(tmp, f"{len(made)}.wav")
                            engine.to_wav(key, p)
                            made[key] = p
                        return made[key]

                # Play on from here without waiting for the window: redrawing and saving the place
                # take longer than the audio buffer, which is what made the gap between passages.
                while k < len(passages):
                    wav = wav_for(k)
                    if k + 1 < len(passages):  # make the next passage while this one plays
                        threading.Thread(target=wav_for, args=(k + 1,), daemon=True).start()
                    if not self.playing or gen != self.gen:
                        break
                    self.events.put(("show", ch, k, gen))
                    if not self.player.play(wav):
                        break
                    k += 1
                else:
                    if self.playing and gen == self.gen:
                        self.events.put(("advance", ch, len(passages) - 1, gen))
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
                elif kind == "show" and ev[3] == self.gen and ev[1] == self.ch:
                    self.highlight(ev[2])  # follow the voice; it has already moved on by itself
                elif kind == "advance" and ev[3] == self.gen and ev[1] == self.ch:
                    self.skip(1)
                elif kind == "status":
                    self.status.set(ev[1])
                elif kind == "error":
                    self.stop()
                    messagebox.showerror("PDF Audiobook", ev[1])
        except queue.Empty:
            pass
        self.after(15, self._pump)

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

    def export_epub(self):
        """Read-along EPUB: the original pages, highlighted as they are read
        (play it in Apple Books with Read Aloud). The speed slider sets the
        narration speed, since Books has no speed control of its own."""
        if not self.book:
            return
        out = filedialog.askdirectory(title="Folder for the read-along EPUB")
        if not out:
            return
        self.stop()
        self.status.set("Exporting read-along EPUB (original pages + highlighted narration)…")

        def work():
            from .epub import export_epub
            from .tts import get_engine

            try:
                path = export_epub(self.pdf, self.book, get_engine(rate=round(self.rate.get(), 2)), out,
                                   log=lambda m: self.events.put(("status", str(m))))
                self.events.put(("status", f"Wrote {path}. Copy it to your iPhone and open it in Books."))
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
    try:  # long imports must not make the rest of the PC sluggish
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
    except Exception:
        pass
    Reader(argv[0] if argv else None).mainloop()


if __name__ == "__main__":
    main()
