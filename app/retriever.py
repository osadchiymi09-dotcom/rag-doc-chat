"""Поиск релевантных чанков по вопросу.

Основной ранжировщик — BM25 (околослучайная классика IR, реализована
с нуля, без тяжёлых зависимостей). Если установлены sentence-transformers,
поверх BM25 включается эмбеддинг-реранкер: семантика ловит синонимы,
которых нет в лексике вопроса.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from .chunker import Chunk

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Разбить текст на слова: нижний регистр, без пунктуации и цифр."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Retriever:
    """Индекс документов + BM25-поиск.

    Пример:
        retriever = Retriever(chunks)
        hits = retriever.search("как называется проект?")
    """
    chunks: list[Chunk]
    _doc_tokens: list[list[str]] = field(init=False, repr=False)
    _doc_freq: Counter = field(init=False, repr=False)
    _avg_len: float = field(init=False, repr=False)
    _n_docs: int = field(init=False, repr=False)
    k1: float = 1.5
    b: float = 0.75
    _embedder: Optional[object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._doc_tokens = [_tokenize(c.text) for c in self.chunks]
        self._n_docs = len(self._doc_tokens)
        self._doc_freq = Counter(
            term for toks in self._doc_tokens for term in set(toks)
        )
        self._avg_len = (
            sum(len(t) for t in self._doc_tokens) / self._n_docs
            if self._n_docs
            else 1.0
        )
        self._embedder = None

    # ---- скоринг -------------------------------------------------

    def _idf(self, term: str) -> float:
        """Обратная частота документа (сглаженная, не уходит в 0)."""
        n = self._doc_freq.get(term, 0)
        return math.log(1 + (self._n_docs - n + 0.5) / (n + 0.5))

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        toks = self._doc_tokens[doc_index]
        if not toks:
            return 0.0
        freq = Counter(toks)
        dl = len(toks)
        score = 0.0
        for term in query_terms:
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_len)
            score += idf * tf * (self.k1 + 1) / denom
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Вернуть топ-k чанков с баллами, убывая по релевантности."""
        terms = _tokenize(query)
        if not terms:
            return []
        scored = [
            (self.chunks[i], self._score(terms, i))
            for i in range(self._n_docs)
        ]
        scored = [s for s in scored if s[1] > min_score]
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:top_k]

    # ---- опциональный реранк через эмбеддинги --------------------

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return
        self._embedder = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    def rerank(self, query: str, hits, top_k: int = 5):
        """Семантический реранк поверх BM25. Если эмбеддингов нет —
        возвращает исходный порядок."""
        if self._embedder is None:
            self._load_embedder()
        if self._embedder is None:
            return hits[:top_k]
        docs = [ch.text for ch, _ in hits]
        q_emb = self._embedder.encode([query], normalize_embeddings=True)
        d_embs = self._embedder.encode(docs, normalize_embeddings=True)
        sims = (d_embs @ q_emb.T).flatten()
        ranked = sorted(
            zip(hits, sims.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(ch, score) for (ch, bm25), score in ranked][:top_k]
