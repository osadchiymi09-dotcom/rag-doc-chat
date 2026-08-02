"""Интерактивный CLI-режим: задавай вопросы без Telegram.

Запуск:
    python -m app.cli --docs docs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generator import Generator
from .loader import count_sources, load_folder
from .retriever import Retriever


def build_pipeline(docs_dir: str):
    """Собрать индекс и вернуть (retriever, generator)."""
    chunks = load_folder(docs_dir)
    if not chunks:
        raise RuntimeError(f"В {docs_dir} нет документов (.txt/.md/.pdf)")
    retriever = Retriever(chunks)
    generator = Generator()
    return retriever, generator


def ask(retriever: Retriever, generator: Generator, question: str) -> str:
    """Один вопрос → строка с ответом и источниками."""
    hits = retriever.search(question, top_k=5)
    if not hits:
        return "Не нашёл ничего по такому вопросу в документах."
    best = [ch for ch, _ in hits]
    answer = generator.answer(question, best)
    out = answer.text
    if answer.sources:
        out += "\n\n📎 " + ", ".join(answer.sources)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Спроси свои документы")
    parser.add_argument("--docs", default="docs", help="папка с документами")
    parser.add_argument("-q", "--question", help="вопрос одной строкой")
    args = parser.parse_args(argv)

    try:
        retriever, generator = build_pipeline(args.docs)
    except (RuntimeError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    stats = count_sources(retriever.chunks)
    total = sum(stats.values())
    print(f"✓ Проиндексировано {total} чанков: "
          f"{', '.join(f'{k} ({v})' for k, v in stats.items())}")

    if args.question:
        print("\n" + ask(retriever, generator, args.question))
        return 0

    print("\nЗадавай вопросы (exit / Ctrl-D — выход):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q or q.lower() in {"exit", "quit", "выход"}:
            return 0
        try:
            print("\n" + ask(retriever, generator, q))
        except Exception as exc:
            print(f"✗ {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
