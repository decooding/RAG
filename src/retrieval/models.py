"""
Модели данных поискового движка (Модуль 3: Retrieval Engine).
Результат поиска SearchResult строго соответствует требованиям DoD.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SearchResult:
    """
    Результат поиска релевантного фрагмента НПА.
    Обязательные поля согласно DoD:
    - text: текст чанка / статьи
    - metadata: словарь метаданных
    - score: оценка релевантности (BM25, косинусная близость или RRF-скор)
    - rank: позиция в итоговой выдаче (1, 2, ...)
    - retrieval_time_ms: время поиска в миллисекундах (замерено через time.perf_counter())
    """
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    retrieval_time_ms: float
    mode: str
    article_number: str = ""
    article_title: str = ""

    def __post_init__(self):
        # Гарантируем извлечение и сохранение связи с номером статьи
        if not self.article_number and "article_number" in self.metadata:
            self.article_number = str(self.metadata["article_number"])
        if not self.article_title and "article_title" in self.metadata:
            self.article_title = str(self.metadata["article_title"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": round(self.score, 6),
            "rank": self.rank,
            "retrieval_time_ms": round(self.retrieval_time_ms, 3),
            "mode": self.mode,
            "article_number": self.article_number,
            "article_title": self.article_title
        }
