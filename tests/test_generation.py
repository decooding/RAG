"""
Автоматизированные тесты для верификации критериев приёмки (DoD)
Модуля 4 (Generation Pipeline): промпт-инжиниринг, параметры модели, замер TTFT и извлечение ссылок.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.config import GENERATION_TEMPERATURE, GENERATION_MAX_OUTPUT_TOKENS, GEMINI_GENERATION_MODEL_NAME
from src.retrieval.models import SearchResult
from src.generation.prompts import LEGAL_SYSTEM_PROMPT, build_user_prompt
from src.generation.llm_client import GeminiLegalGenerator, GenerationResult, ARTICLE_CITATION_PATTERN


class TestDoDModule4(unittest.TestCase):

    def test_dod_m4_parameters_and_config(self):
        """DoD М4: Зафиксированы параметры temperature=0.1, max_output_tokens=1024, модель gemini-1.5-flash."""
        generator = GeminiLegalGenerator(api_key="mock_key")
        self.assertEqual(generator.temperature, 0.1)
        self.assertEqual(generator.max_output_tokens, 1024)
        self.assertEqual(generator.model_name, "gemini-1.5-flash")
        self.assertEqual(GENERATION_TEMPERATURE, 0.1)
        self.assertEqual(GENERATION_MAX_OUTPUT_TOKENS, 1024)

    def test_dod_m4_system_prompt_legal_rules(self):
        """DoD М4: Системный промпт содержит строгий регламент (только предоставленные статьи, цитирование, запрет додумывания)."""
        prompt = LEGAL_SYSTEM_PROMPT
        self.assertIn("ИСКЛЮЧИТЕЛЬНО на основе положений статей", prompt)
        self.assertIn("ЗАПРЕТ ДОДУМЫВАНИЯ", prompt)
        self.assertIn("Данный вопрос не регулируется предоставленными статьями", prompt)
        self.assertIn("ОБЯЗАТЕЛЬНОЕ ЦИТИРОВАНИЕ", prompt)

    def test_dod_m4_build_user_prompt_with_context(self):
        """DoD М4: Контекст формируется со всеми номерами статей и метаданными."""
        chunks = [
            SearchResult(
                id="doc_1",
                text="Текст первой статьи о земельном фонде.",
                metadata={"source": "k030000442_.08-09-2026.rus.pdf", "article_number": "1", "article_title": "Земельный фонд"},
                score=0.9,
                rank=1,
                retrieval_time_ms=10.0,
                mode="dense",
                article_number="1",
                article_title="Земельный фонд"
            )
        ]
        prompt = build_user_prompt(query="Какие категории земель?", search_results=chunks)
        self.assertIn("Статья 1: Земельный фонд", prompt)
        self.assertIn("Текст первой статьи о земельном фонде", prompt)
        self.assertIn("Какие категории земель?", prompt)

    def test_dod_m4_citation_extraction(self):
        """DoD М4: Корректное извлечение ссылок на статьи (включая дефисные номера 14-1, 24)."""
        generator = GeminiLegalGenerator(api_key="mock_key")
        answer = (
            "Согласно статье 1 Земельного кодекса РК, земельный фонд делится на категории. "
            "В соответствии со статьей 24-1 иностранные граждане имеют ограничения. "
            "Также см. ст. 107."
        )
        citations = generator.extract_citations(answer)
        self.assertIn("1", citations)
        self.assertIn("24-1", citations)
        self.assertIn("107", citations)

    def test_dod_m4_streaming_ttft_and_latency_measurement(self):
        """DoD М4: Замер TTFT и полной задержки через стриминг и time.perf_counter()."""
        generator = GeminiLegalGenerator(api_key="mock_key")

        # Мокируем вызов client.models.generate_content_stream
        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "Согласно статье 1 "
        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "земли делятся на 7 категорий."

        mock_stream = [mock_chunk_1, mock_chunk_2]

        generator._client = MagicMock()
        generator._client.models.generate_content_stream.return_value = mock_stream

        chunks = [
            SearchResult(
                id="doc_1",
                text="Земельный фонд",
                metadata={"article_number": "1", "article_title": "Фонд"},
                score=0.9,
                rank=1,
                retrieval_time_ms=5.0,
                mode="bm25"
            )
        ]

        result = generator.generate(query="Категории?", search_results=chunks)

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.answer, "Согласно статье 1 земли делятся на 7 категорий.")
        self.assertIn("1", result.citations)
        self.assertGreater(result.ttft_ms, 0.0)
        self.assertGreaterEqual(result.total_ms, result.ttft_ms)

        # Проверка формата to_dict() согласно критериям DoD
        d = result.to_dict()
        for key in ["answer", "citations", "ttft_ms", "total_ms"]:
            self.assertIn(key, d, f"Ключ {key} отсутствует в ответе DoD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
