"""
Единый фасад поискового движка (Модуль 3: Retrieval Engine).
Предоставляет стандартный интерфейс:
    retrieve(query: str, top_k: int = 5, mode: str = "hybrid") -> List[SearchResult]

Реализует три режима:
- 'bm25': только лексический поиск
- 'dense': семантический поиск через ChromaDB
- 'hybrid': гибридный поиск (RRF слияние)
"""

import sys
import argparse
from typing import List, Optional, Dict, Any
from src.config import COLLECTION_STRUCTURAL_LOCAL, BM25_STRUCTURAL_PATH
from src.retrieval.models import SearchResult
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.dense_search import DenseSearcher
from src.retrieval.hybrid_search import HybridSearcher


class RAGSearchEngine:
    """Универсальный фасад поискового движка RAG."""

    def __init__(
        self,
        collection_name: str = COLLECTION_STRUCTURAL_LOCAL,
        bm25_path=BM25_STRUCTURAL_PATH
    ):
        self.collection_name = collection_name
        self.bm25_searcher = BM25Searcher(index_path=bm25_path)
        self.dense_searcher = DenseSearcher(collection_name=collection_name)
        self.hybrid_searcher = HybridSearcher(
            dense_searcher=self.dense_searcher,
            bm25_searcher=self.bm25_searcher
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Главный интерфейс поиска.
        Критерии DoD:
        1. При пустом или бессмысленном запросе возвращает [], не бросая исключений.
        2. Время поиска замеряется строго через time.perf_counter() и сохраняется в SearchResult.
        3. Для всех чанков гарантированно сохраняется связь со статьей (article_number).
        """
        # Защита от пустых/некорректных/бессмысленных запросов (DoD)
        if not query or not isinstance(query, str) or not query.strip():
            return []

        from src.indexing.bm25_indexer import tokenize_legal_text
        if not tokenize_legal_text(query):
            return []

        mode_clean = mode.lower().strip()

        if mode_clean == "bm25":
            return self.bm25_searcher.search(query=query, top_k=top_k)
        elif mode_clean == "dense":
            return self.dense_searcher.search(query=query, top_k=top_k, where=where)
        elif mode_clean == "hybrid":
            return self.hybrid_searcher.search(query=query, top_k=top_k, where=where)
        else:
            raise ValueError(f"Неизвестный режим поиска: '{mode}'. Доступны: 'bm25', 'dense', 'hybrid'.")


# Глобальный синглтон для удобного вызова функции
_DEFAULT_ENGINE: Optional[RAGSearchEngine] = None


def retrieve(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    where: Optional[Dict[str, Any]] = None
) -> List[SearchResult]:
    """
    Функциональный интерфейс поиска строго по спецификации DoD:
    retrieve(query: str, top_k: int, mode: str) -> List[SearchResult]
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = RAGSearchEngine()
    return _DEFAULT_ENGINE.retrieve(query=query, top_k=top_k, mode=mode, where=where)


def main():
    """CLI для интерактивного тестирования поискового движка."""
    parser = argparse.ArgumentParser(description="Поиск по статьям Кодексов РК (BM25 / Dense / Hybrid)")
    parser.add_argument("query", nargs="?", default="категории земель сельскохозяйственного назначения", help="Текст поискового запроса")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Количество результатов (Top-K)")
    parser.add_argument("--mode", "-m", choices=["bm25", "dense", "hybrid"], default="hybrid", help="Режим поиска")

    args = parser.parse_args()

    print(f"Поисковый запрос: '{args.query}'")
    print(f"Режим: {args.mode.upper()}, Top-K: {args.top_k}")
    print("=" * 60)

    engine = RAGSearchEngine()
    results = engine.retrieve(query=args.query, top_k=args.top_k, mode=args.mode)

    if not results:
        print("Ничего не найдено (или пустой запрос).")
        return

    print(f"Найдено результатов: {len(results)}")
    for res in results:
        print(f"\n[Ранг #{res.rank}] Статья: {res.article_number} | Заголовок: {res.article_title}")
        print(f"  Скор ({res.mode}): {res.score:.5f} | Задержка поиска: {res.retrieval_time_ms:.2f} мс")
        print(f"  Фрагмент текста: {res.text[:150].replace(chr(10), ' ')}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
