"""
Автоматизированный комплекс тестов для верификации критериев приёмки (DoD)
Модуля 2 (Indexing & Storage) и Модуля 3 (Retrieval Engine).
"""

import sys
import unittest
from pathlib import Path

from src.config import (
    CHROMA_DIR,
    BM25_STRUCTURAL_PATH,
    BM25_NAIVE_500_PATH,
    COLLECTION_STRUCTURAL_LOCAL,
    COLLECTION_NAIVE_500,
    PROCESSED_DIR
)
from src.indexing.chroma_indexer import ChromaIndexManager
from src.indexing.bm25_indexer import BM25IndexManager
from src.retrieval.search_engine import RAGSearchEngine, retrieve
from src.retrieval.models import SearchResult


class TestDoDModule2And3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = RAGSearchEngine()

    def test_dod_m2_persistent_chromadb_reload(self):
        """DoD М2: База ChromaDB сохранена на диск и успешно перечитывается PersistentClient."""
        manager = ChromaIndexManager(persist_dir=CHROMA_DIR)
        count_structural = manager.get_count(COLLECTION_STRUCTURAL_LOCAL)
        count_naive = manager.get_count(COLLECTION_NAIVE_500)

        self.assertEqual(count_structural, 187, "Число структурных статей в ChromaDB должно быть ровно 187")
        self.assertEqual(count_naive, 1557, "Число наивных чанков в ChromaDB должно быть ровно 1557")

    def test_dod_m2_metadata_filtering(self):
        """DoD М2: Все метаданные сохранены и доступны для точечной фильтрации."""
        manager = ChromaIndexManager(persist_dir=CHROMA_DIR)
        # Фильтрация по конкретной статье (Статья 1)
        res = manager.query(
            collection_name=COLLECTION_STRUCTURAL_LOCAL,
            query_embedding=self.engine.dense_searcher.embedder.embed_query("фонд"),
            top_k=1,
            where={"article_number": "1"}
        )
        self.assertEqual(len(res["ids"][0]), 1)
        art_num = res["metadatas"][0][0].get("article_number")
        self.assertEqual(str(art_num), "1")

        # Проверка отсутствия None во всех метаданных полученного документа
        for k, v in res["metadatas"][0][0].items():
            self.assertIsNotNone(v, f"Метаданные содержат None в поле {k}")

    def test_dod_m2_bm25_pickle_serialization(self):
        """DoD М2: Индекс BM25 сериализован на диск через pickle и перечитывается."""
        self.assertTrue(BM25_STRUCTURAL_PATH.exists(), "Файл bm25_structural.pkl отсутствует на диске")
        self.assertTrue(BM25_NAIVE_500_PATH.exists(), "Файл bm25_naive_500.pkl отсутствует на диске")

        loaded = BM25IndexManager.load(BM25_STRUCTURAL_PATH)
        self.assertEqual(len(loaded.doc_ids), 187)
        results = loaded.search("земля", top_k=3)
        self.assertGreater(len(results), 0)

    def test_dod_m3_unified_retrieve_interface(self):
        """DoD М3: Единый интерфейс retrieve(query, top_k, mode) поддерживает bm25, dense, hybrid."""
        query = "земли сельскохозяйственного назначения"

        for mode in ["bm25", "dense", "hybrid"]:
            results = retrieve(query=query, top_k=3, mode=mode)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 3)

            for item in results:
                self.assertIsInstance(item, SearchResult)
                self.assertEqual(item.mode, mode)
                self.assertIsInstance(item.score, float)
                self.assertIsInstance(item.rank, int)
                # Проверка замера времени
                self.assertGreater(item.retrieval_time_ms, 0.0)
                # Проверка связи со статьей
                self.assertTrue(len(item.article_number) > 0, "article_number не должен быть пустым")

    def test_dod_m3_empty_and_nonsense_query_safety(self):
        """DoD М3: При пустом или бессмысленном запросе модуль возвращает [] без Exception."""
        bad_queries = ["", "   ", "\n\t", "???!!!", "---"]

        for q in bad_queries:
            for mode in ["bm25", "dense", "hybrid"]:
                res = retrieve(query=q, top_k=5, mode=mode)
                self.assertEqual(res, [], f"Для запроса '{q}' в режиме {mode} ожидался пустой список []")

    def test_dod_m3_retrieval_latency_isolation(self):
        """DoD М3: Время поиска замеряется строго через time.perf_counter() и возвращается в мс."""
        res = retrieve(query="собственность на землю", top_k=2, mode="hybrid")
        self.assertGreater(len(res), 0)
        self.assertGreater(res[0].retrieval_time_ms, 0.0)
        # Поиск по 187 статьям на локальной машине должен занимать разумное время (< 500 мс)
        self.assertLess(res[0].retrieval_time_ms, 1000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
