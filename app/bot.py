"""Telegram-бот: спроси свои документы текстом или голосом.

Работает через Bot API (токен от @BotFather), поэтому запускается
без привязки к пользовательскому аккаунту. Голосовые расшифровываются
локально whisper.cpp (см. transcribe.py).

Запуск:
    TELEGRAM_BOT_TOKEN=... python -m app.bot --docs docs/
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

from .cli import ask, build_pipeline
from .loader import count_sources
from .transcribe import transcribe

API = "https://api.telegram.org/bot{token}"


class RAGBot:
    def __init__(self, token: str, docs_dir: str) -> None:
        self.token = token
        self.base = API.format(token=token)
        self.retriever, self.generator = build_pipeline(docs_dir)
        self.offset = 0

    # ---- низкий уровень ------------------------------------------

    def _call(self, method: str, **kwargs):
        resp = requests.post(f"{self.base}/{method}", **kwargs, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"TG API {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("result")

    def send(self, chat_id, text: str):
        self._call(
            "sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
        )

    def get_updates(self):
        return self._call(
            "getUpdates",
            json={"offset": self.offset, "timeout": 30},
        ) or []

    def _file_bytes(self, file_id: str) -> bytes:
        info = self._call("getFile", json={"file_id": file_id})
        path = info["file_path"]
        resp = requests.get(
            f"https://api.telegram.org/file/bot{self.token}/{path}",
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content

    # ---- обработка сообщений --------------------------------------

    def handle(self, msg: dict) -> None:
        chat_id = msg["chat"]["id"]
        text = msg.get("text") or ""
        voice = msg.get("voice")
        if text.startswith("/start"):
            stats = count_sources(self.retriever.chunks)
            n = sum(stats.values())
            self.send(
                chat_id,
                f"Привет! Вопросы к моим документам — текстом или голосом.\n"
                f"Индекс: {len(stats)} файлов, {n} чанков.",
            )
            return
        if voice:
            file_id = voice["file_id"]
            self.send(chat_id, "🎙️ Слушаю...")
            try:
                blob = self._file_bytes(file_id)
                path = os.path.join(
                    os.environ.get("TMPDIR", "/tmp"), f"tg_voice_{chat_id}.ogg"
                )
                with open(path, "wb") as f:
                    f.write(blob)
                question = transcribe(path)
                if not question:
                    self.send(chat_id, "Не разобрал голос, попробуй ещё раз.")
                    return
                self.send(chat_id, f"🎙️ «{question}»")
                reply = ask(self.retriever, self.generator, question)
            except Exception as exc:
                reply = f"✗ Ошибка голоса: {exc}"
            self.send(chat_id, reply)
            return
        if text:
            self.send(chat_id, "🔍 Ищу ответ...")
            try:
                reply = ask(self.retriever, self.generator, text)
            except Exception as exc:
                reply = f"✗ {exc}"
            self.send(chat_id, reply)

    # ---- цикл ------------------------------------------------------

    def run(self) -> None:
        print("Бот запущен. Жду сообщения...")
        while True:
            for upd in self.get_updates():
                self.offset = upd["update_id"] + 1
                msg = upd.get("message")
                if not msg:
                    continue
                try:
                    self.handle(msg)
                except Exception as exc:
                    print(f"Ошибка: {exc}", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TG-бот для RAG по документам")
    parser.add_argument("--docs", default="docs", help="папка с документами")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    args = parser.parse_args(argv)
    if not args.token:
        print("Нужен TELEGRAM_BOT_TOKEN (от @BotFather)", file=sys.stderr)
        return 1
    try:
        RAGBot(args.token, args.docs).run()
    except (RuntimeError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
