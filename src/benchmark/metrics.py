"""
Модуль расчета академических метрик информационного поиска и оценки RAG (Модуль 5).
Включает:
- Hit@K: индикатор попадания целевой статьи в Top-K выдачи (0 или 1);
- MRR (Mean Reciprocal Rank): обратный ранг первого вхождения целевой статьи (1 / rank);
- Precision@K: точность в окне K (число релевантных / K);
- Recall@K: полнота в окне K (число найденных релевантных / общее число релевантных);
- Citation Accuracy: проверка присутствия ссылки на целевую статью в ответе LLM.
"""

import re
from typing import List, Union
from src.generation.llm_client import ARTICLE_CITATION_PATTERN


def normalize_article_num(art: Union[str, int]) -> str:
    """Нормализует номер статьи: удаляет слово 'статья', пробелы, приводит к нижнему регистру."""
    s = str(art).strip().lower()
    s = re.sub(r'^(?:статья|ст\.?)\s*', '', s)
    return s.strip()


def hit_at_k(retrieved_articles: List[str], target_article: str, k: int) -> int:
    """
    Hit@K: возвращает 1, если целевая статья присутствует среди первых K результатов, иначе 0.
    """
    if not retrieved_articles or k <= 0:
        return 0

    target_norm = normalize_article_num(target_article)
    top_k_items = [normalize_article_num(a) for a in retrieved_articles[:k]]

    return 1 if target_norm in top_k_items else 0


def mrr(retrieved_articles: List[str], target_article: str) -> float:
    """
    Reciprocal Rank (RR): обратный ранг первого вхождения целевой статьи (1.0 / rank).
    Если статья не найдена, возвращает 0.0.
    """
    if not retrieved_articles:
        return 0.0

    target_norm = normalize_article_num(target_article)
    for rank_idx, article in enumerate(retrieved_articles, start=1):
        if normalize_article_num(article) == target_norm:
            return 1.0 / rank_idx

    return 0.0


def precision_at_k(retrieved_articles: List[str], target_article: str, k: int) -> float:
    """
    Precision@K: доля релевантных документов среди первых K результатов.
    В задаче с 1 целевой статьей: 1/K если найдена, иначе 0.0.
    """
    if not retrieved_articles or k <= 0:
        return 0.0

    hit = hit_at_k(retrieved_articles, target_article, k)
    return float(hit) / float(k)


def recall_at_k(retrieved_articles: List[str], target_article: str, k: int) -> float:
    """
    Recall@K: доля найденных релевантных документов от общего числа релевантных (1).
    В задаче с 1 целевой статьей: 1.0 если найдена, иначе 0.0.
    """
    if not retrieved_articles or k <= 0:
        return 0.0

    return float(hit_at_k(retrieved_articles, target_article, k))


def citation_accuracy(answer: str, target_article: str) -> bool:
    """
    Citation Accuracy: проверяет, процитирован ли целевой номер статьи в ответе LLM.
    Учитывает все падежи (статье, статьей, ст.) и прямое вхождение номера в юридическом контексте.
    """
    if not answer or not target_article:
        return False

    target_norm = normalize_article_num(target_article)
    found_citations = [
        normalize_article_num(c)
        for c in ARTICLE_CITATION_PATTERN.findall(answer)
    ]

    if target_norm in found_citations:
        return True

    # Дополнительная проверка на точное упоминание: "статья 1", "статье 1", "ст. 1"
    explicit_pattern = re.compile(
        rf'(?:стать[еияюей]|ст\.?)\s*{re.escape(target_norm)}(?![0-9-])',
        re.IGNORECASE
    )
    if explicit_pattern.search(answer):
        return True

    return False
