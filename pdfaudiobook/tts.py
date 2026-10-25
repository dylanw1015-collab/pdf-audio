"""Text-to-speech backends and audio playback.

Backends, best quality first:
* piper   - local neural TTS built on VITS (Kim, Kong & Son, ICML 2021, whose
            listening test "achieves a MOS comparable to ground truth").
            Runs offline on a CPU faster than real time.
* system  - the OS voice through pyttsx3 (SAPI5 on Windows, NSSpeechSynthesizer
            on macOS, eSpeak on Linux). No download, quality depends on the OS.
* espeak  - eSpeak NG formant synthesis. Very fast and tiny, clearly robotic.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave

_SENT_RE = re.compile(r"(?<=[.!?”\"])\s+(?=[A-Z“\"'(])")


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split into sentence groups no longer than max_chars.

    Neural TTS models are trained on sentence-length utterances; very long
    inputs drift in prosody, so we feed them a few sentences at a time.
    """
    out: list[str] = []
    for para in text.split("\n\n"):
        cur = ""
        for sent in _SENT_RE.split(para.strip()):
            if len(cur) + len(sent) + 1 > max_chars and cur:
                out.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            out.append(cur)
    return out


def default_voice_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".pdfaudiobook", "voices")


class Piper:
    name = "piper"

    def __init__(self, voice: str = "en_US-lessac-medium", rate: float = 1.0):
        from piper import PiperVoice, SynthesisConfig

        path = voice if voice.endswith(".onnx") else os.path.join(default_voice_dir(), voice + ".onnx")
        if not os.path.exists(path):
            os.makedirs(default_voice_dir(), exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "piper.download_voices", "--download-dir", default_voice_dir(), voice],
                check=True,
            )
        self.voice = PiperVoice.load(path)
        # length_scale > 1 is slower speech, so invert the user's rate.
        self.cfg = SynthesisConfig(length_scale=1.0 / rate)

    def to_wav(self, text: str, path: str) -> None:
        with wave.open(path, "wb") as wf:
            self.voice.synthesize_wav(text, wf, syn_config=self.cfg)


class Espeak:
    name = "espeak"

    def __init__(self, voice: str = "en-us", rate: float = 1.0):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise RuntimeError("espeak-ng is not installed")
        self.voice, self.wpm = voice, int(175 * rate)

    def to_wav(self, text: str, path: str) -> None:
        subprocess.run([self.exe, "-v", self.voice, "-s", str(self.wpm), "-w", path, text], check=True)


class System:
    name = "system"

    def __init__(self, voice: str | None = None, rate: float = 1.0):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", int(self.engine.getProperty("rate") * rate))
        if voice:
            self.engine.setProperty("voice", voice)

    def to_wav(self, text: str, path: str) -> None:
        self.engine.save_to_file(text, path)
        self.engine.runAndWait()


def get_engine(name: str = "auto", voice: str | None = None, rate: float = 1.0):
    order = [name] if name != "auto" else ["piper", "system", "espeak"]
    errors = []
    for n in order:
        try:
            if n == "piper":
                return Piper(voice or "en_US-lessac-medium", rate)
            if n == "espeak":
                return Espeak(voice or "en-us", rate)
            if n == "system":
                return System(voice, rate)
        except Exception as e:  # try the next backend
            errors.append(f"{n}: {e}")
    raise RuntimeError("No TTS backend worked:\n  " + "\n  ".join(errors))


def concat_wavs(parts: list[str], out: str, gap_s: float = 0.35) -> float:
    """Join WAVs (same format) with a short pause between; return seconds."""
    with wave.open(parts[0]) as w:
        params = w.getparams()
    silence = b"\x00" * int(params.framerate * gap_s) * params.sampwidth * params.nchannels
    frames = 0
    with wave.open(out, "wb") as o:
        o.setparams(params)
        for p in parts:
            with wave.open(p) as w:
                data = w.readframes(w.getnframes())
            o.writeframes(data + silence)
            frames += len(data + silence) // (params.sampwidth * params.nchannels)
    return frames / params.framerate


def play_wav(path: str) -> None:
    """Play a WAV and block until it ends, using what the OS already has."""
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME)
        return
    for cmd in (["afplay"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd + [path], check=False)
            return
    raise RuntimeError("No audio player found (install ffmpeg for ffplay).")


def synth_chunks(engine, chunks: list[str], tmpdir: str | None = None) -> list[str]:
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="pdfab_")
    paths = []
    for i, c in enumerate(chunks):
        p = os.path.join(tmpdir, f"{i:05d}.wav")
        engine.to_wav(c, p)
        paths.append(p)
    return paths
