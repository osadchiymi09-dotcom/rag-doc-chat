"""Генерация ответа по найденным чанкам через LLM.

Используется OpenRouter (OpenAI-совместимый API) с бесплатными
моделями по умолчанию. Модель и ключ задаются через переменные
окружения OPENROUTER_API_KEY / RAG_MODEL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from .chunker import Chunk

DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 90

SYSTEM_PROMPT = (
    "Ты — помощник, который отвечает строго по предоставленным "
    "фрагментам документов. Если в фрагментах нет ответа — так и скажи. "
    "Отвечай на русском, коротко и по делу. В конце перечисли источники "
    "в формате [файл] (без длинных цитат)."
)


@dataclass
class RAGAnswer:
    text: str
    sources: list[str]
    used_chunks: list[Chunk]
    model: str


class Generator:
    """Пишет ответ по контексту через OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("RAG_MODEL", DEFAULT_MODEL)
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY не задан — кинь ключ в .env или окружение"
            )

    @staticmethod
    def _build_context(
        chunks: list[Chunk], max_chars: int = 6000
    ) -> tuple[str, list[Chunk]]:
        """Склеить чанки в контекст с разметкой источников.

        Возвращает (контекст, реально использованные чанки) — источники
        должны отражать именно то, что попало в промпт.
        """
        blocks = []
        used_chunks: list[Chunk] = []
        total = 0
        for ch in chunks:
            block = f"[ИСТОЧНИК: {ch.source}]\n{ch.text}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            used_chunks.append(ch)
            total += len(block)
        return "\n\n".join(blocks), used_chunks

    def answer(
        self,
        question: str,
        chunks: list[Chunk],
        max_context_chars: int = 6000,
    ) -> RAGAnswer:
        """Спросить LLM по чанкам, вернуть ответ + источники."""
        context, used_chunks = self._build_context(chunks, max_context_chars)
        user_msg = (
            f"ВОПРОС:\n{question}\n\n"
            f"ДОКУМЕНТЫ:\n{context}\n\n"
            "Ответь по документам и перечисли источники [файл]."
        )
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.2,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM ответил {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        sources = list(dict.fromkeys(ch.source for ch in used_chunks))
        return RAGAnswer(
            text=text,
            sources=sources,
            used_chunks=used_chunks,
            model=self.model,
        )
