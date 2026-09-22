"""
Модуль лексического индекса на базе BM25Okapi (библиотека rank_bm25).
Включает юридический токенизатор и сериализацию/десериализацию через pickle для быстрого старта.
"""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from src.data.models import ArticleChunk


# Регулярное выражение для токенизации:
# 1. Токены с дефисами (например '14-1', '270-4', 'нормативно-правовой')
# 2. Обычные буквенно-цифровые слова
TOKEN_PATTERN = re.compile(r'[a-zA-Zа-яА-ЯёЁ0-9]+(?:-[a-zA-Zа-яА-ЯёЁ0-9]+)*')


def tokenize_legal_text(text: str) -> List[str]:
    """Токенизация юридического текста: нижний регистр, сохранение номеров статей с дефисами."""
    if not text:
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25IndexManager:
    """Управление лексическим индексом BM25 и сериализацией на диск."""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[ArticleChunk]):
        """Построение индекса BM25 по списку чанков."""
        self.doc_ids = [c.id for c in chunks]
        self.documents = [c.text for c in chunks]
        self.metadatas = [c.metadata for c in chunks]

        print(f"Токенизация {len(chunks)} документов для BM25...")
        corpus_tokens = [tokenize_legal_text(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        print("[OK] Индекс BM25 успешно построен.")

    def save(self, filepath: Path):
        """Сериализация индекса BM25 и метаданных в pickle-файл."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self.bm25,
            "doc_ids": self.doc_ids,
            "documents": self.documents,
            "metadatas": self.metadatas
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Индекс BM25 сохранен в: {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path) -> "BM25IndexManager":
        """Быстрая десериализация индекса BM25 с диска без повторной токенизации."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Файл индекса BM25 не найден: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.bm25 = data["bm25"]
        instance.doc_ids = data["doc_ids"]
        instance.documents = data["documents"]
        instance.metadatas = data["metadatas"]
        return instance

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Поиск по индексу BM25.
        Возвращает список кортежей: (doc_id, text, metadata, score).
        """
        query_tokens = tokenize_legal_text(query)
        if not query_tokens or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append((
                self.doc_ids[idx],
                self.documents[idx],
                self.metadatas[idx],
                score
            ))
        return results
