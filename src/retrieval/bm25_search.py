"""
Лексический поиск (BM25) в Модуле 3.
Использует сериализованный индекс с диска, замеряет время строго через time.perf_counter().
"""

import time
from pathlib import Path
from typing import List, Optional
from src.config import BM25_STRUCTURAL_PATH
from src.indexing.bm25_indexer import BM25IndexManager
from src.retrieval.models import SearchResult


class BM25Searcher:
    """Компонент лексического поиска по сохраненному индексу BM25."""

    def __init__(self, index_path: Path = BM25_STRUCTURAL_PATH):
        self.index_path = Path(index_path)
        self.indexer = BM25IndexManager.load(self.index_path)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Поиск по запросу.
        При пустом или бессмысленном запросе возвращает пустой список (DoD).
        """
        if not query or not query.strip():
            return []

        # Замер времени строго через time.perf_counter()
        t_start = time.perf_counter()
        raw_results = self.indexer.search(query=query, top_k=top_k)
        retrieval_time_ms = (time.perf_counter() - t_start) * 1000.0

        search_results: List[SearchResult] = []
        for rank, (doc_id, text, metadata, score) in enumerate(raw_results, start=1):
            search_results.append(
                SearchResult(
                    id=doc_id,
                    text=text,
                    metadata=metadata,
                    score=score,
                    rank=rank,
                    retrieval_time_ms=retrieval_time_ms,
                    mode="bm25"
                )
            )

        return search_results
