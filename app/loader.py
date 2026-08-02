"""Загрузка документов из папки в список чанков.

Поддерживает .txt / .md / .rst нативно и .pdf через pypdf
(опциональная зависимость — если пакет не установлен, PDF
пропускается с предупреждением).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .chunker import Chunk, chunk_text

SUPPORTED_TEXT = {".txt", ".md", ".rst", ".log"}
PDF_EXT = ".pdf"


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            f"PDF-файл {path.name}: нужен пакет pypdf — "
            "`pip install pypdf`"
        )
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(path: Path, **chunk_kwargs) -> list[Chunk]:
    """Прочитать один файл и нарезать его на чанки."""
    ext = path.suffix.lower()
    if ext in SUPPORTED_TEXT:
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif ext == PDF_EXT:
        text = _read_pdf(path)
    else:
        return []
    return chunk_text(text, source=path.name, **chunk_kwargs)


def load_folder(
    folder: str | Path,
    recursive: bool = True,
    **chunk_kwargs,
) -> list[Chunk]:
    """Собрать чанки из всех поддерживаемых файлов в папке.

    Аргументы:
        folder: путь к папке с документами.
        recursive: заходить ли во вложенные папки.
        chunk_kwargs: параметры чанкера (chunk_size, overlap).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} не папка")
    pattern = "**/*" if recursive else "*"
    chunks: list[Chunk] = []
    skipped: list[str] = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_TEXT | {PDF_EXT}:
            continue
        try:
            chunks.extend(load_document(path, **chunk_kwargs))
        except ImportError as exc:
            skipped.append(str(exc))
        except Exception as exc:  # кривой файл не должен ронять всё
            skipped.append(f"{path.name}: {exc}")
    if skipped and not chunks:
        raise RuntimeError("; ".join(skipped))
    return chunks


def count_sources(chunks: Iterable[Chunk]) -> dict[str, int]:
    """Сколько чанков пришлось на каждый документ (для отчёта)."""
    counts: dict[str, int] = {}
    for ch in chunks:
        counts[ch.source] = counts.get(ch.source, 0) + 1
    return counts
