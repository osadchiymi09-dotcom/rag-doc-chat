"""Разбиение документов на чанки с перекрытием.

Стратегия: текст режется на абзацы, абзацы склеиваются в чанки
целевого размера, между соседними чанками остаётся перекрытие
(overlap), чтобы вопрос, ответ на который лежит на стыке,
не потерялся.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Один фрагмент документа."""
    text: str
    source: str            # путь к файлу-источнику
    index: int             # номер чанка внутри документа
    chars: int             # длина в символах


def _split_paragraphs(text: str) -> list[str]:
    """Абзацы = куски текста между пустыми строками."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[Chunk]:
    """Нарезать текст на чанки размером ~chunk_size с перекрытием overlap.

    Аргументы:
        text: исходный текст документа.
        source: имя файла (для обратной ссылки в ответе).
        chunk_size: целевой размер чанка в символах.
        overlap: перекрытие между чанками в символах.
    """
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buf = ""
    for para in paragraphs:
        # Абзац больше чанка — режем его самого на куски.
        if len(para) > chunk_size:
            if buf:
                chunks.append(Chunk(buf, source, len(chunks), len(buf)))
                buf = ""
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i : i + chunk_size].strip()
                if piece:
                    chunks.append(Chunk(piece, source, len(chunks), len(piece)))
            continue
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(Chunk(buf, source, len(chunks), len(buf)))
            # Хвост текущего буфера попадает в начало следующего чанка
            # (перекрытие), чтобы не терять контекст на стыке.
            tail = buf[-overlap:] if len(buf) >= overlap else ""
            buf = f"{tail}\n{para}".strip()
    if buf:
        chunks.append(Chunk(buf, source, len(chunks), len(buf)))
    return chunks
