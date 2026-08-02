"""Тесты ядра: чанкер, загрузка, BM25-поиск. Без внешних зависимостей."""
import unittest
from pathlib import Path

from app.chunker import chunk_text
from app.loader import load_document, load_folder
from app.retriever import Retriever

TEXT = (
    "Компания «ЛунаСофт» делает чат-ботов.\n"
    "Она основана в 2019 году в Петербурге.\n"
    "\n"
    "Джуниор-разработчик получает от 80 до 120 тысяч рублей.\n"
    "Стажировка длится три месяца.\n"
    "\n"
    "Телеграм-бот обрабатывает до 5000 сообщений в день.\n"
    "Написать в отдел кадров: hr@lunasoft.ru.\n"
)


class TestChunker(unittest.TestCase):
    def test_overlap_in_merge_branch(self):
        # Абзацы меньше chunk_size, но сумма больше — склейка с перекрытием.
        # Хвост предыдущего чанка должен попасть в начало следующего.
        paras = [
            f"Абзац номер {i}: " + "данные " * 8 + f"маркер{i}"
            for i in range(4)
        ]
        text = "\n\n".join(paras)
        chunks = chunk_text(text, "t.md", chunk_size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        tail = chunks[0].text[-20:]
        self.assertIn(tail, chunks[1].text)

    def test_long_paragraph_cut_into_pieces(self):
        # Абзац длиннее chunk_size режется на перекрывающиеся куски.
        para = "слово" * 100  # 400 символов
        chunks = chunk_text(para, "t.md", chunk_size=120, overlap=30)
        self.assertGreaterEqual(len(chunks), 3)
        joined = " ".join(c.text for c in chunks)
        self.assertIn("слово", joined)

    def test_paragraph_order_preserved(self):
        chunks = chunk_text(TEXT, "t.md")
        joined = " ".join(c.text for c in chunks)
        self.assertIn("2019", joined)
        self.assertIn("hr@lunasoft.ru", joined)

    def test_chunk_has_source_and_index(self):
        chunks = chunk_text(TEXT, "guide.md", chunk_size=2000, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "guide.md")
        self.assertEqual(chunks[0].index, 0)


class TestLoader(unittest.TestCase):
    def test_load_md(self):
        tmp = Path("/tmp") / "rag_test_doc.md"
        tmp.write_text(TEXT, encoding="utf-8")
        try:
            chunks = load_document(tmp)
            self.assertTrue(chunks)
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_folder_skips_unknown(self):
        tmp = Path("/tmp") / "rag_test_folder"
        tmp.mkdir(exist_ok=True)
        (tmp / "a.md").write_text(TEXT, encoding="utf-8")
        (tmp / "b.py").write_text("print(1)", encoding="utf-8")
        try:
            chunks = load_folder(tmp)
            self.assertEqual({c.source for c in chunks}, {"a.md"})
        finally:
            (tmp / "a.md").unlink(missing_ok=True)
            (tmp / "b.py").unlink(missing_ok=True)
            tmp.rmdir()


class TestRetriever(unittest.TestCase):
    def setUp(self):
        chunks = chunk_text(TEXT, "t.md")
        self.r = Retriever(chunks)

    def test_relevant_top(self):
        hits = self.r.search("сколько получает джуниор")
        self.assertTrue(hits)
        top_text = hits[0][0].text.lower()
        self.assertIn("80", top_text)

    def test_no_hits(self):
        self.assertEqual(self.r.search("квантовая физика ыц", top_k=3), [])

    def test_topk_limit(self):
        hits = self.r.search("документ текст чат", top_k=2)
        self.assertLessEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
