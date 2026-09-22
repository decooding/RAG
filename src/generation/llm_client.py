"""
LLM Клиент на базе Google Gemini 1.5 Flash (Модуль 4: Generation Pipeline).
Использует официальный SDK google-genai, стриминг для фиксации TTFT (Time-to-First-Token)
и замера полной задержки total_ms строго через time.perf_counter().
Включает exponential backoff при сетевых ошибках и rate limits.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from src.config import (
    GEMINI_GENERATION_MODEL_NAME,
    GENERATION_TEMPERATURE,
    GENERATION_MAX_OUTPUT_TOKENS,
    GEMINI_API_KEY
)
from src.retrieval.models import SearchResult
from src.generation.prompts import LEGAL_SYSTEM_PROMPT, build_user_prompt


# Регулярное выражение для поиска упоминаний статей во всех падежах русского языка (например: "статье 1", "статьей 24-1", "ст. 3", "статьях 5")
ARTICLE_CITATION_PATTERN = re.compile(
    r'(?:стать(?:ями|ям|ях|ей|ёй|[яиею])|ст\.?)\s*([0-9]+(?:-[0-9]+)*[а-яА-Яa-zA-Z]?)',
    re.IGNORECASE
)


@dataclass
class GenerationResult:
    """Структурированный результат генерации согласно DoD Модуля 4."""
    answer: str
    citations: List[str]
    ttft_ms: float
    total_ms: float
    model: str = GEMINI_GENERATION_MODEL_NAME

    def to_dict(self) -> Dict[str, Any]:
        """Строгое соответствие DoD: answer, citations, ttft_ms, total_ms."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "model": self.model
        }


class GeminiLegalGenerator:
    """Генератор юридических ответов с поддержкой TTFT и retry на Gemini 1.5 Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GEMINI_GENERATION_MODEL_NAME,
        temperature: float = GENERATION_TEMPERATURE,
        max_output_tokens: int = GENERATION_MAX_OUTPUT_TOKENS,
        system_instruction: str = LEGAL_SYSTEM_PROMPT
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_instruction = system_instruction
        self._client = None

        if self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def extract_citations(
        self,
        answer_text: str,
        provided_results: Optional[List[Union[SearchResult, Dict[str, Any]]]] = None
    ) -> List[str]:
        """
        Извлекает номера статей, упомянутых в ответе, и сопоставляет их с контекстом.
        """
        found_in_text = ARTICLE_CITATION_PATTERN.findall(answer_text)
        citations = set(found_in_text)

        # Если в предоставленных результатах есть статьи, проверяем прямое упоминание их номеров
        if provided_results:
            for item in provided_results:
                art_num = item.article_number if isinstance(item, SearchResult) else item.get("article_number", "")
                if art_num and (art_num in citations or f" {art_num}" in answer_text):
                    citations.add(str(art_num))

        return sorted(list(citations), key=lambda x: [int(p) if p.isdigit() else p for p in x.split("-")])

    def generate(
        self,
        query: str,
        search_results: List[Union[SearchResult, Dict[str, Any]]],
        max_retries: int = 4
    ) -> GenerationResult:
        """
        Генерирует юридический ответ по запросу и списку найденных статей.
        Замеряет:
        - ttft_ms: задержка до первого полученного токена/чанка со стрима.
        - total_ms: общее время генерации ответа.
        При ошибках rate limit использует экспоненциальный откат (backoff).
        """
        if not self.is_available:
            raise RuntimeError(
                "GEMINI_API_KEY не установлен. Укажите валидный API-ключ в переменных окружения "
                "или передайте в конструктор GeminiLegalGenerator(api_key=...)"
            )

        from google.genai import types

        user_prompt = build_user_prompt(query=query, search_results=search_results)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            system_instruction=self.system_instruction,
        )

        delay = 1.0
        for attempt in range(max_retries):
            try:
                # Фиксация точки старта запроса (строго time.perf_counter())
                t_start = time.perf_counter()
                ttft_ms = 0.0
                chunks: List[str] = []

                # Потоковый вызов для измерения TTFT
                stream_response = self._client.models.generate_content_stream(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config
                )

                for chunk in stream_response:
                    if ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    if chunk.text:
                        chunks.append(chunk.text)

                total_ms = (time.perf_counter() - t_start) * 1000.0
                if ttft_ms == 0.0:
                    ttft_ms = total_ms

                answer = "".join(chunks).strip()
                citations = self.extract_citations(answer, search_results)

                return GenerationResult(
                    answer=answer,
                    citations=citations,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    model=self.model_name
                )

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit_or_network = any(k in err_str for k in ["429", "rate limit", "quota", "connection", "timeout"])
                if is_rate_limit_or_network and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise RuntimeError(f"Ошибка вызова Gemini 1.5 Flash API (попытка {attempt+1}/{max_retries}): {e}")

        raise RuntimeError("Не удалось получить ответ от Gemini API после повторных попыток.")
