"""
Гибридный поиск (Hybrid Search) в Модуле 3.
Объединяет результаты Dense и BM25 поиска с использованием метода Reciprocal Rank Fusion (RRF).
Замеряет общее время поиска строго через time.perf_counter().
"""

import time
from typing import List, Optional, Dict, Any
from collections import defaultdict
from src.config import RRF_K
from src.retrieval.models import SearchResult
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.dense_search import DenseSearcher


class HybridSearcher:
    """Компонент гибридного поиска с ранжированием по Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_searcher: DenseSearcher,
        bm25_searcher: BM25Searcher,
        rrf_k: int = RRF_K
    ):
        self.dense_searcher = dense_searcher
        self.bm25_searcher = bm25_searcher
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Гибридный поиск слиянием рангов Dense и BM25.
        При пустом запросе возвращает пустой список без исключений (DoD).
        """
        if not query or not query.strip():
            return []

        # Замер общего времени поиска через time.perf_counter()
        t_start = time.perf_counter()

        candidates_k = max(top_k * candidate_multiplier, 15)

        # Выполняем оба поиска
        dense_results = self.dense_searcher.search(query=query, top_k=candidates_k, where=where)
        bm25_results = self.bm25_searcher.search(query=query, top_k=candidates_k)

        # Если оба поиска вернули пустоту
        if not dense_results and not bm25_results:
            return []

        # Словарь для агрегации RRF-скора и хранения данных документов
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_store: Dict[str, Dict[str, Any]] = {}

        # 1. Учет плотных (Dense) рангов
        for res in dense_results:
            doc_id = res.id
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + res.rank)
            if doc_id not in doc_store:
                doc_store[doc_id] = {
                    "text": res.text,
                    "metadata": res.metadata,
                    "article_number": res.article_number,
                    "article_title": res.article_title
                }

        # 2. Учет лексических (BM25) рангов
        for res in bm25_results:
            doc_id = res.id
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + res.rank)
            if doc_id not in doc_store:
                doc_store[doc_id] = {
                    "text": res.text,
                    "metadata": res.metadata,
                    "article_number": res.article_number,
                    "article_title": res.article_title
                }

        # Сортировка по убыванию RRF-скора
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        total_retrieval_time_ms = (time.perf_counter() - t_start) * 1000.0

        hybrid_results: List[SearchResult] = []
        for rank, doc_id in enumerate(sorted_doc_ids, start=1):
            info = doc_store[doc_id]
            hybrid_results.append(
                SearchResult(
                    id=doc_id,
                    text=info["text"],
                    metadata=info["metadata"],
                    score=rrf_scores[doc_id],
                    rank=rank,
                    retrieval_time_ms=total_retrieval_time_ms,
                    mode="hybrid",
                    article_number=info["article_number"],
                    article_title=info["article_title"]
                )
            )

        return hybrid_results
