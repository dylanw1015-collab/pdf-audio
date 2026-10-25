"""Write chapters to audio files: one MP3 per chapter and/or one M4B audiobook.

M4B is an MP4/AAC file with a chapter table, which audiobook players (Apple
Books, BookPlayer, Smart AudioBook Player, VLC) show as a chapter list. The
chapter table is written with ffmpeg's FFMETADATA format
(https://ffmpeg.org/ffmpeg-formats.html#Metadata-2).

Bitrates: speech is narrow-band compared with music, so 64 kbps mono MP3 and
48 kbps mono AAC are transparent enough for a voice and keep a 10-hour book
under ~300 MB.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from .chapters import spoken_text
from .math import MathSpeaker
from .tts import chunk_text, concat_wavs, synth_chunks


def safe_name(s: str, n: int = 60) -> str:
    return re.sub(r"[^\w\- ]+", "", s).strip().replace(" ", "_")[:n] or "chapter"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is needed for MP3/M4B export (WAV export works without it).")
    return exe


def chapter_wav(engine, speaker, ch, out_wav: str) -> float:
    chunks = chunk_text(speaker.speak_text(spoken_text(ch)))
    with tempfile.TemporaryDirectory(prefix="pdfab_") as tmp:
        parts = synth_chunks(engine, chunks, tmp)
        return concat_wavs(parts, out_wav)


def export_book(book: dict, engine, out_dir: str, fmt: str = "mp3", chapters=None, log=print) -> list[str]:
    """fmt: "wav", "mp3", "m4b", or "mp3+m4b". Returns written paths."""
    os.makedirs(out_dir, exist_ok=True)
    speaker = MathSpeaker()
    chs = book["chapters"]
    idx = chapters or range(1, len(chs) + 1)
    written, wavs = [], []
    work = tempfile.mkdtemp(prefix="pdfab_wav_")
    try:
        for n in idx:
            ch = chs[n - 1]
            log(f"[{n}/{len(chs)}] {ch.title}")
            wav = os.path.join(out_dir if fmt == "wav" else work, f"{n:02d}_{safe_name(ch.title)}.wav")
            dur = chapter_wav(engine, speaker, ch, wav)
            wavs.append((n, ch.title, wav, dur))
            if fmt == "wav":
                written.append(wav)
            if "mp3" in fmt:
                mp3 = os.path.join(out_dir, os.path.basename(wav)[:-4] + ".mp3")
                subprocess.run(
                    [_ffmpeg(), "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-b:a", "64k",
                     "-metadata", f"title={ch.title}", "-metadata", f"album={book['title']}",
                     "-metadata", f"track={n}/{len(chs)}", mp3],
                    check=True,
                )
                written.append(mp3)
        if "mp3" in fmt:
            m3u = os.path.join(out_dir, safe_name(book["title"]) + ".m3u")
            with open(m3u, "w", encoding="utf-8") as f:
                f.write("#EXTM3U\n")
                for (n, t, w, d), p in zip(wavs, [w for w in written if w.endswith(".mp3")]):
                    f.write(f"#EXTINF:{int(d)},{t}\n{os.path.basename(p)}\n")
            written.append(m3u)
        if "m4b" in fmt:
            written.append(_write_m4b(book, wavs, out_dir))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return written


def _write_m4b(book: dict, wavs, out_dir: str) -> str:
    out = os.path.join(out_dir, safe_name(book["title"]) + ".m4b")
    with tempfile.TemporaryDirectory(prefix="pdfab_m4b_") as tmp:
        lst = os.path.join(tmp, "list.txt")
        meta = os.path.join(tmp, "meta.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for _, _, w, _ in wavs:
                f.write("file '" + w.replace("'", "'\\''") + "'\n")
        t = 0
        with open(meta, "w", encoding="utf-8") as f:
            esc = lambda s: re.sub(r"([=;#\\\n])", r"\\\1", s)
            f.write(f";FFMETADATA1\ntitle={esc(book['title'])}\nalbum={esc(book['title'])}\ngenre=Audiobook\n")
            for _, title, _, d in wavs:
                ms = int(d * 1000)
                f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={t}\nEND={t + ms}\ntitle={esc(title)}\n")
                t += ms
        subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
             "-i", meta, "-map_metadata", "1", "-map_chapters", "1",
             "-ac", "1", "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", out],
            check=True,
        )
    return out
