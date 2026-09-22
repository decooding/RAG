"""
Конфигурация проекта RAG по законодательству Республики Казахстан.
Содержит пути, имена коллекций, параметры моделей эмбеддингов и поисковых алгоритмов.
"""

from pathlib import Path
import os

# Корень проекта (папка d:/RAG)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папки для данных и индексов
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
INDICES_DIR = DATA_DIR / "indices"

# Создание необходимых каталогов
for folder in [DATA_DIR, PROCESSED_DIR, CHROMA_DIR, INDICES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Имена коллекций ChromaDB
COLLECTION_STRUCTURAL_LOCAL = "rk_articles_dense_local"
COLLECTION_STRUCTURAL_GEMINI = "rk_articles_dense_gemini"
COLLECTION_NAIVE_500 = "rk_articles_naive_500"

# Пути к сериализованным индексам BM25
BM25_STRUCTURAL_PATH = INDICES_DIR / "bm25_structural.pkl"
BM25_NAIVE_500_PATH = INDICES_DIR / "bm25_naive_500.pkl"

# Модели эмбеддингов и генерации
LOCAL_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
GEMINI_EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_GENERATION_MODEL_NAME = "gemini-1.5-flash"

# Параметры генерации (DoD Модуль 4)
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_OUTPUT_TOKENS = 1024

# Параметры пакетирования и алгоритмов
INDEXING_BATCH_SIZE = 64
RRF_K = 60  # Константа сглаживания для Reciprocal Rank Fusion

# API Ключи (поддерживаем оба варианта переменной окружения)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
