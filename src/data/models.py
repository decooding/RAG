"""
Модели данных для чанков нормативно-правовых актов (НПА РК).
Гарантирует строгую типизацию и отсутствие значений None в метаданных (требование DoD).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Union


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    Приводит все значения словаря к типам str, int, float, bool.
    Заменяет None на пустую строку или безопасное дефолтное значение для совместимости с ChromaDB.
    """
    sanitized: Dict[str, Union[str, int, float, bool]] = {}
    for key, val in meta.items():
        if val is None:
            sanitized[key] = ""
        elif isinstance(val, bool):
            sanitized[key] = val
        elif isinstance(val, (int, float, str)):
            sanitized[key] = val
        else:
            sanitized[key] = str(val)
    return sanitized


@dataclass
class ArticleChunk:
    """
    Представление чанка статьи Кодекса РК или наивного текстового чанка.
    """
    id: str
    text: str
    article_number: str
    article_title: str
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def __post_init__(self):
        # Гарантируем санитизацию метаданных и отсутствие None
        self.metadata = sanitize_metadata(self.metadata)
        # Обязательно синхронизируем ключевые поля с метаданными для фильтрации
        self.metadata["article_number"] = str(self.article_number)
        self.metadata["article_title"] = str(self.article_title)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "article_number": self.article_number,
            "article_title": self.article_title,
            "metadata": dict(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticleChunk":
        meta = dict(data.get("metadata", {}))
        art_num = str(data.get("article_number", meta.get("article_number", "")))
        art_title = str(data.get("article_title", meta.get("article_title", "")))
        chunk_id = str(data.get("id", f"art_{art_num}"))
        return cls(
            id=chunk_id,
            text=data.get("text", ""),
            article_number=art_num,
            article_title=art_title,
            metadata=meta
        )
