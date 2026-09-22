"""
Юнит-тесты для Модуля 5 (Benchmark & Evaluation).
Проверяют математическую корректность Hit@K, MRR, Precision@K, Recall@K,
citation_accuracy и структуру вывода бенчмарка.
"""

import unittest
from pathlib import Path
from src.benchmark.metrics import (
    hit_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    citation_accuracy,
    normalize_article_num
)


class TestBenchmarkMetrics(unittest.TestCase):

    def test_normalize_article_num(self):
        self.assertEqual(normalize_article_num("1"), "1")
        self.assertEqual(normalize_article_num("Статья 1"), "1")
        self.assertEqual(normalize_article_num("ст. 24-1"), "24-1")
        self.assertEqual(normalize_article_num("  14-1  "), "14-1")

    def test_hit_at_k(self):
        retrieved = ["1", "24", "107", "97", "2"]

        # Hit@1
        self.assertEqual(hit_at_k(retrieved, "1", k=1), 1)
        self.assertEqual(hit_at_k(retrieved, "24", k=1), 0)

        # Hit@3
        self.assertEqual(hit_at_k(retrieved, "24", k=3), 1)
        self.assertEqual(hit_at_k(retrieved, "107", k=3), 1)
        self.assertEqual(hit_at_k(retrieved, "97", k=3), 0)

        # Отсутствующая статья
        self.assertEqual(hit_at_k(retrieved, "999", k=5), 0)

        # Пустые списки
        self.assertEqual(hit_at_k([], "1", k=5), 0)

    def test_mrr(self):
        retrieved = ["1", "24", "107", "97"]

        # Ранг 1 -> MRR = 1/1 = 1.0
        self.assertAlmostEqual(mrr(retrieved, "1"), 1.0)

        # Ранг 2 -> MRR = 1/2 = 0.5
        self.assertAlmostEqual(mrr(retrieved, "24"), 0.5)

        # Ранг 3 -> MRR = 1/3 ≈ 0.33333
        self.assertAlmostEqual(mrr(retrieved, "107"), 1.0 / 3.0)

        # Не найдена -> 0.0
        self.assertEqual(mrr(retrieved, "500"), 0.0)

    def test_precision_and_recall_at_k(self):
        retrieved = ["1", "24", "107"]

        # Статья 1 на первом месте
        self.assertAlmostEqual(precision_at_k(retrieved, "1", k=1), 1.0)
        self.assertAlmostEqual(recall_at_k(retrieved, "1", k=1), 1.0)

        self.assertAlmostEqual(precision_at_k(retrieved, "1", k=3), 1.0 / 3.0)
        self.assertAlmostEqual(recall_at_k(retrieved, "1", k=3), 1.0)

        # Статья 24 на втором месте
        self.assertAlmostEqual(precision_at_k(retrieved, "24", k=1), 0.0)
        self.assertAlmostEqual(recall_at_k(retrieved, "24", k=1), 0.0)

        self.assertAlmostEqual(precision_at_k(retrieved, "24", k=2), 1.0 / 2.0)
        self.assertAlmostEqual(recall_at_k(retrieved, "24", k=2), 1.0)

    def test_citation_accuracy(self):
        # Положительные сценарии с разными падежами и написанием
        ans1 = "В соответствии со статьей 1 Земельного кодекса РК земли делятся на 7 категорий."
        self.assertTrue(citation_accuracy(ans1, "1"))

        ans2 = "Согласно статье 24-1 иностранцы не имеют права частной собственности."
        self.assertTrue(citation_accuracy(ans2, "24-1"))

        ans3 = "Норма определена в ст. 107 Земельного кодекса."
        self.assertTrue(citation_accuracy(ans3, "107"))

        # Отрицательные сценарии
        ans_neg = "Данный вопрос не регулируется законодательством."
        self.assertFalse(citation_accuracy(ans_neg, "1"))

        ans_wrong = "Порядок регулируется статьей 84 Земельного кодекса."
        self.assertFalse(citation_accuracy(ans_wrong, "1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
