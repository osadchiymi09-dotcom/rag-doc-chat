"""Транскрибация голосовых через локальный whisper.cpp.

whisper.cpp не умеет читать opus/ogg, поэтому файл сначала
конвертируется в WAV (afconvert на macOS / ffmpeg на остальных).
Модель и бинарник настраиваются через окружение.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

WHISPER_CLI = os.getenv("WHISPER_CLI", "whisper-cli")
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "~/models/whisper/ggml-small.bin",
)
WHISPER_LANG = os.getenv("WHISPER_LANG", "ru")


def _to_wav(src: str, dst: str) -> None:
    """Ogg/opus → wav. На macOS — afconvert, иначе ffmpeg."""
    if shutil.which("afconvert"):
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", src, dst],
            check=True, capture_output=True,
        )
        return
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
            check=True, capture_output=True,
        )
        return
    raise RuntimeError("Нужен afconvert (macOS) или ffmpeg для конвертации ogg→wav")


def transcribe(audio_path: str) -> str:
    """Вернуть распознанный текст из аудиофайла (.ogg/.wav/.mp3...)."""
    if not shutil.which(WHISPER_CLI):
        raise RuntimeError(
            f"whisper-cli не найден ({WHISPER_CLI}). "
            "Установи: https://github.com/ggerganov/whisper.cpp"
        )
    model = os.path.expanduser(WHISPER_MODEL)
    if not os.path.exists(model):
        raise FileNotFoundError(f"Нет модели whisper: {model}")

    tmp_dir = tempfile.mkdtemp(prefix="rag_voice_")
    wav = os.path.join(tmp_dir, "voice.wav")
    try:
        if not audio_path.lower().endswith((".wav", ".mp3")):
            _to_wav(audio_path, wav)
            audio_path = wav
        out = subprocess.run(
            [WHISPER_CLI, "-m", model, "-f", audio_path,
             "-l", WHISPER_LANG, "-otxt", "-of",
             os.path.join(tmp_dir, "out")],
            capture_output=True,
        )
        txt_path = os.path.join(tmp_dir, "out.txt")
        if not os.path.exists(txt_path):
            err = out.stderr.decode(errors="ignore")[-400:]
            raise RuntimeError(f"whisper не вернул текст: {err}")
        with open(txt_path, encoding="utf-8") as f:
            return f.read().strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
