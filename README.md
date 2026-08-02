# 🎙️ rag-doc-chat — спрашивай свои документы голосом

RAG-помощник: закидываешь папку с документами, задаёшь вопросы **текстом или голосом** — отвечает нейросеть по твоим документам, а не по памяти, и показывает источники.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Что внутри

- **Поиск BM25 с нуля** — без тяжёлых ML-зависимостей, индексация за секунды. Плюс опциональный семантический реранк через sentence-transformers (ставится по желанию).
- **Чанкер с перекрытием** — абзацы склеиваются в чанки ~1500 символов, между соседними чанками остаётся контекст, чтобы не терять ответ на стыке.
- **Ответы с цитатами** — LLM (Gemma 4 на OpenRouter, бесплатно) отвечает строго по фрагментам и перечисляет файлы-источники.
- **Голосовой ввод** — голосовые расшифровываются **локально** whisper.cpp (без отправки аудио в облако).
- **Форматы** — TXT, MD, RST, LOG, PDF (PDF через `pypdf`).
- **Два интерфейса** — интерактивный CLI и Telegram-бот (обычный Bot API, любой токен от @BotFather).

## Демо

```
$ python -m app.cli --docs docs/ -q "Нужен ли опыт для стажировки?"
✓ Проиндексировано 3 чанков: api_guide.md (1), company.md (2)

Нет, стажировка рассчитана на студентов и выпускников без опыта.
Важны базовые знания Python и желание учиться.

[company.md]

📎 company.md
```

> 🔑 **Ключ у каждого свой (BYOK).** В репозитории нет и не будет ничьих ключей. Каждый пользователь вписывает **свой** `OPENROUTER_API_KEY` в `.env` — в `.env.example` только заглушка. Никогда не коммить `.env` и ключи.

## Установка

```bash
git clone https://github.com/osadchiymi09-dotcom/rag-doc-chat
cd rag-doc-chat
pip install -r requirements.txt            # pypdf — только если нужен PDF
cp .env.example .env                       # впиши СВОЙ OPENROUTER_API_KEY
python -m app.cli --docs docs/             # CLI-режим
```

**Голос в Telegram:** поставь whisper.cpp (`whisper-cli`) и модель `ggml-small.bin`, задай `TELEGRAM_BOT_TOKEN` в `.env`, запусти `python -m app.bot --docs docs/`.

## Структура

```
app/
├── chunker.py     # нарезка текста на чанки с перекрытием
├── loader.py      # чтение .txt/.md/.pdf из папки
├── retriever.py   # BM25 + опциональный реранк эмбеддингами
├── generator.py   # ответ LLM по контексту с источниками
├── transcribe.py  # локальная расшифровка голоса (whisper.cpp)
├── cli.py         # интерактивный CLI
└── bot.py         # Telegram-бот (текст + голос)
```

## Тесты

```bash
python -m unittest discover tests -v
```

## Идеи дальше

- Подсветка точных цитат в тексте ответа
- Авто-индексирование новых файлов по хоткею `/reindex`
- Реранк по умолчанию (включить sentence-transformers в requirements)

---

Сделано на Python 3.9+, без одного тяжёлого ML-стека. Лицензия MIT.
