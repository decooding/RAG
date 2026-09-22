"""
Полный пайплайн RAG (Поиск + Генерация).
Объединяет Модуль 3 (Retrieval Engine) и Модуль 4 (Generation Pipeline).
Формирует юридически строгий ответ и замеряет все ключевые академические метрики:
retrieval_latency_ms, ttft_ms, generation_latency_ms, e2e_latency_ms.
"""

import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.retrieval.search_engine import RAGSearchEngine, retrieve
from src.retrieval.models import SearchResult
from src.generation.llm_client import GeminiLegalGenerator, GenerationResult


@dataclass
class RAGAnswer:
    """Итоговый юридический ответ RAG-системы со всеми метаданными и замерами задержек."""
    query: str
    answer: str
    citations: List[str]
    retrieved_articles: List[Dict[str, Any]]
    retrieval_time_ms: float
    ttft_ms: float
    generation_time_ms: float
    e2e_latency_ms: float
    search_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_articles": self.retrieved_articles,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "e2e_latency_ms": round(self.e2e_latency_ms, 2),
            "search_mode": self.search_mode
        }


class LegalRAGPipeline:
    """Комплексный пайплайн поиска и генерации юридических ответов."""

    def __init__(
        self,
        search_engine: Optional[RAGSearchEngine] = None,
        llm_client: Optional[GeminiLegalGenerator] = None
    ):
        self.search_engine = search_engine or RAGSearchEngine()
        self.llm_client = llm_client or GeminiLegalGenerator()

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "hybrid"
    ) -> RAGAnswer:
        """
        Выполняет полный цикл RAG:
        1. Поиск релевантных статей (BM25 / Dense / Hybrid).
        2. Формирование замкнутого контекста.
        3. Стриминг через Gemini 1.5 Flash с замером TTFT и генерации.
        4. Расчет суммарного E2E Latency.
        """
        # Шаг 1: Поиск
        search_results: List[SearchResult] = self.search_engine.retrieve(
            query=query,
            top_k=top_k,
            mode=search_mode
        )

        retrieval_time_ms = search_results[0].retrieval_time_ms if search_results else 0.0

        # Шаг 2: Генерация
        gen_result: GenerationResult = self.llm_client.generate(
            query=query,
            search_results=search_results
        )

        e2e_latency = retrieval_time_ms + gen_result.total_ms

        retrieved_brief = [
            {
                "article_number": r.article_number,
                "article_title": r.article_title,
                "score": r.score,
                "rank": r.rank
            }
            for r in search_results
        ]

        return RAGAnswer(
            query=query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            retrieved_articles=retrieved_brief,
            retrieval_time_ms=retrieval_time_ms,
            ttft_ms=gen_result.ttft_ms,
            generation_time_ms=gen_result.total_ms,
            e2e_latency_ms=e2e_latency,
            search_mode=search_mode
        )


def main():
    parser = argparse.ArgumentParser(description="Юридический RAG ассистент по Кодексам РК")
    parser.add_argument("query", nargs="?", default="Какие категории земель существуют в Республике Казахстан?", help="Вопрос пользователя")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Количество статей контекста")
    parser.add_argument("--mode", "-m", choices=["bm25", "dense", "hybrid"], default="hybrid", help="Режим поиска")

    args = parser.parse_args()

    pipeline = LegalRAGPipeline()
    if not pipeline.llm_client.is_available:
        print("[ВНИМАНИЕ] Переменная GEMINI_API_KEY не задана в окружении.")
        print("Для работы генератора установите ключ: $env:GEMINI_API_KEY='ваш_ключ'")
        return

    print(f"Вопрос: {args.query}")
    print(f"Режим поиска: {args.mode.upper()}, Контекст: Top-{args.top_k} статей")
    print("=" * 60)

    rag_answer = pipeline.answer_query(
        query=args.query,
        top_k=args.top_k,
        search_mode=args.mode
    )

    print("\n--- ОТВЕТ ЮРИДИЧЕСКОГО АССИСТЕНТА ---")
    print(rag_answer.answer)
    print("\n--- МЕТАДАННЫЕ И ТАЙМИНГИ (DoD) ---")
    print(f"Использованные статьи (Citations): {rag_answer.citations}")
    print(f"Время поиска (Retrieval):        {rag_answer.retrieval_time_ms:.2f} мс")
    print(f"Время первого токена (TTFT):     {rag_answer.ttft_ms:.2f} мс")
    print(f"Время генерации (Total Gen):     {rag_answer.generation_time_ms:.2f} мс")
    print(f"Полная задержка (E2E Latency):   {rag_answer.e2e_latency_ms:.2f} мс")
    print("=" * 60)


if __name__ == "__main__":
    main()
