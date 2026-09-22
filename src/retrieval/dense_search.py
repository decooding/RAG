"""
Семантический поиск (Dense Retrieval) в Модуле 3.
Использует персистентную ChromaDB и модель эмбеддингов.
Замеряет время строго через time.perf_counter().
"""

import time
from typing import List, Optional, Dict, Any
from src.config import COLLECTION_STRUCTURAL_LOCAL
from src.indexing.embedder import BaseEmbedder, LocalEmbedder
from src.indexing.chroma_indexer import ChromaIndexManager
from src.retrieval.models import SearchResult


class DenseSearcher:
    """Компонент векторного семантического поиска через ChromaDB."""

    def __init__(
        self,
        collection_name: str = COLLECTION_STRUCTURAL_LOCAL,
        embedder: Optional[BaseEmbedder] = None,
        chroma_manager: Optional[ChromaIndexManager] = None
    ):
        self.collection_name = collection_name
        self.embedder = embedder or LocalEmbedder()
        self.chroma_manager = chroma_manager or ChromaIndexManager()

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Семантический поиск по вектору запроса.
        При пустом или бессмысленном запросе возвращает пустой список (DoD).
        """
        # При пустом или бессмысленном запросе (нет букв/цифр) возвращаем пустой список (DoD)
        if not query or not isinstance(query, str) or not query.strip():
            return []

        from src.indexing.bm25_indexer import tokenize_legal_text
        if not tokenize_legal_text(query):
            return []

        # Замер времени строго через time.perf_counter()
        t_start = time.perf_counter()

        # Векторизация запроса с префиксом query:
        query_vector = self.embedder.embed_query(query)

        # Поиск в ChromaDB
        response = self.chroma_manager.query(
            collection_name=self.collection_name,
            query_embedding=query_vector,
            top_k=top_k,
            where=where
        )

        retrieval_time_ms = (time.perf_counter() - t_start) * 1000.0

        search_results: List[SearchResult] = []
        if not response or not response.get("ids") or not response["ids"][0]:
            return []

        ids = response["ids"][0]
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        distances = response["distances"][0]

        for rank, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            # Преобразуем косинусную дистанцию в оценку сходства: similarity = 1 - distance
            similarity_score = 1.0 - dist
            search_results.append(
                SearchResult(
                    id=doc_id,
                    text=text,
                    metadata=meta,
                    score=similarity_score,
                    rank=rank,
                    retrieval_time_ms=retrieval_time_ms,
                    mode="dense"
                )
            )

        return search_results
