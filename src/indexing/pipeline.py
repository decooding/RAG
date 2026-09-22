"""
Главный пайплайн индексации документов (Модуль 2):
1. Построение и сохранение индекса BM25 (постатейный + наивный).
2. Векторизация и сохранение в персистентную ChromaDB:
   - rk_articles_dense_local (E5-small)
   - rk_articles_naive_500 (E5-small)
   - rk_articles_dense_gemini (text-embedding-004, если доступен API-ключ)
3. Проверка критериев DoD (число документов, фильтрация).
"""

import json
from pathlib import Path
from typing import List
from src.config import (
    PROCESSED_DIR,
    COLLECTION_STRUCTURAL_LOCAL,
    COLLECTION_STRUCTURAL_GEMINI,
    COLLECTION_NAIVE_500,
    BM25_STRUCTURAL_PATH,
    BM25_NAIVE_500_PATH,
    GEMINI_API_KEY
)
from src.data.models import ArticleChunk
from src.indexing.embedder import LocalEmbedder, GeminiEmbedder
from src.indexing.chroma_indexer import ChromaIndexManager
from src.indexing.bm25_indexer import BM25IndexManager


def load_chunks_from_json(path: Path) -> List[ArticleChunk]:
    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    return [ArticleChunk.from_dict(item) for item in raw_list]


def run_indexing_pipeline():
    print("==================================================")
    print(" Запуск пайплайна индексации (Модуль 2: Indexing) ")
    print("==================================================")

    # 1. Загрузка подготовленных чанков
    structural_file = PROCESSED_DIR / "rk_articles_structural.json"
    naive_file = PROCESSED_DIR / "rk_articles_naive_500.json"

    if not structural_file.exists():
        raise FileNotFoundError(f"Файл {structural_file} не найден. Сначала выполните prepare_datasets.py")

    structural_chunks = load_chunks_from_json(structural_file)
    print(f"Загружено постатейных чанков: {len(structural_chunks)}")

    naive_chunks = load_chunks_from_json(naive_file) if naive_file.exists() else []
    print(f"Загружено наивных чанков: {len(naive_chunks)}")

    # 2. Построение и сериализация BM25 индексов
    print("\n--- [1/3] Построение индексов BM25 ---")
    bm25_struct = BM25IndexManager()
    bm25_struct.build_index(structural_chunks)
    bm25_struct.save(BM25_STRUCTURAL_PATH)

    if naive_chunks:
        bm25_naive = BM25IndexManager()
        bm25_naive.build_index(naive_chunks)
        bm25_naive.save(BM25_NAIVE_500_PATH)

    # 3. Инициализация ChromaDB PersistentClient
    print("\n--- [2/3] Векторизация и индексация в ChromaDB ---")
    chroma_mgr = ChromaIndexManager()

    # Локальный эмбеддер (multilingual-e5-small)
    print("Инициализация LocalEmbedder (intfloat/multilingual-e5-small)...")
    local_embedder = LocalEmbedder()

    # Индексация постатейной коллекции
    chroma_mgr.index_chunks(
        collection_name=COLLECTION_STRUCTURAL_LOCAL,
        chunks=structural_chunks,
        embedder=local_embedder,
        batch_size=64,
        recreate=True
    )

    # Индексация наивной коллекции
    if naive_chunks:
        chroma_mgr.index_chunks(
            collection_name=COLLECTION_NAIVE_500,
            chunks=naive_chunks,
            embedder=local_embedder,
            batch_size=64,
            recreate=True
        )

    # Gemini эмбеддер (text-embedding-004), если задан ключ
    if GEMINI_API_KEY:
        print("\nИнициализация GeminiEmbedder (text-embedding-004)...")
        gemini_embedder = GeminiEmbedder()
        chroma_mgr.index_chunks(
            collection_name=COLLECTION_STRUCTURAL_GEMINI,
            chunks=structural_chunks,
            embedder=gemini_embedder,
            batch_size=50,
            recreate=True
        )
    else:
        print("\n[INFO] GEMINI_API_KEY не обнаружен. Коллекция 'rk_articles_dense_gemini' пропущена.")

    # 4. Проверка критериев DoD
    print("\n--- [3/3] Валидация критериев приёмки (DoD) ---")
    struct_count = chroma_mgr.get_count(COLLECTION_STRUCTURAL_LOCAL)
    assert struct_count == len(structural_chunks), (
        f"DoD FAIL: Число документов в ChromaDB ({struct_count}) != числу чанков ({len(structural_chunks)})"
    )
    print(f"  [DoD PASS] Число документов в ChromaDB строго совпадает: {struct_count} == {len(structural_chunks)}")

    # Проверка фильтрации по номеру статьи
    test_query_art = "1"
    query_emb = local_embedder.embed_query("земельный фонд")
    res = chroma_mgr.query(
        collection_name=COLLECTION_STRUCTURAL_LOCAL,
        query_embedding=query_emb,
        top_k=1,
        where={"article_number": test_query_art}
    )
    assert len(res["ids"][0]) > 0, "DoD FAIL: Фильтрация по article_number не вернула результатов"
    retrieved_art_num = res["metadatas"][0][0]["article_number"]
    assert str(retrieved_art_num) == test_query_art, (
        f"DoD FAIL: Ожидалась статья {test_query_art}, получена {retrieved_art_num}"
    )
    print(f"  [DoD PASS] Метаданные доступны для точечной фильтрации: where={{'article_number': '{test_query_art}'}} работает корректно!")

    # Проверка персистентности: пересоздаем клиент и проверяем, что данные на месте
    reloaded_mgr = ChromaIndexManager()
    assert reloaded_mgr.get_count(COLLECTION_STRUCTURAL_LOCAL) == len(structural_chunks), (
        "DoD FAIL: Данные ChromaDB не сохранились на диск при перезапуске клиента"
    )
    print("  [DoD PASS] PersistentClient успешно перечитал базу с диска без повторной векторизации.")

    print("\n[SUCCESS] Модуль 2 (Indexing & Storage) успешно завершил работу и прошел все проверки DoD!")


if __name__ == "__main__":
    run_indexing_pipeline()
