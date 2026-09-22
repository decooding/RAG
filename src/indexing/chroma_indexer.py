"""
Модуль персистентного хранения и векторизации в ChromaDB.
Обеспечивает сохранение на диск (PersistentClient), батч-загрузку по 50-100 документов,
санитизацию метаданных (отсутствие None) и проверку соответствия количества чанков.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR, INDEXING_BATCH_SIZE
from src.data.models import ArticleChunk, sanitize_metadata
from src.indexing.embedder import BaseEmbedder


class ChromaIndexManager:
    """Менеджер персистентного векторного индекса ChromaDB."""

    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # PersistentClient гарантирует сохранение на диск и перечитывание при перезапуске
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(self, name: str, distance_func: str = "cosine"):
        """Получение или создание изолированной коллекции."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance_func}
        )

    def recreate_collection(self, name: str, distance_func: str = "cosine"):
        """Пересоздание коллекции для «чистой» индексации."""
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass
        return self.client.create_collection(
            name=name,
            metadata={"hnsw:space": distance_func}
        )

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[ArticleChunk],
        embedder: BaseEmbedder,
        batch_size: int = INDEXING_BATCH_SIZE,
        recreate: bool = True
    ):
        """
        Пакетная индексация чанков с генерацией эмбеддингов и записью в ChromaDB.
        DoD: Строгое совпадение количества документов и сохранение всех метаданных.
        """
        if recreate:
            collection = self.recreate_collection(collection_name)
        else:
            collection = self.get_or_create_collection(collection_name)

        total_chunks = len(chunks)
        print(f"Индексация коллекции '{collection_name}': всего {total_chunks} документов, батч = {batch_size}...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_ids = [c.id for c in batch]
            batch_texts = [c.text for c in batch]
            # Гарантируем отсутствие None и строгое сохранение article_number
            batch_metas = [sanitize_metadata(c.metadata) for c in batch]

            # Вычисление эмбеддингов
            embeddings = embedder.embed_documents(batch_texts)

            # Пакетное добавление в ChromaDB
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metas
            )
            print(f"  Загружено {min(i + batch_size, total_chunks)} / {total_chunks} документов...")

        final_count = collection.count()
        if final_count != total_chunks:
            raise RuntimeError(
                f"Ошибка индексации: количество документов в коллекции ({final_count}) "
                f"не совпадает с числом чанков ({total_chunks})!"
            )
        print(f"[OK] Коллекция '{collection_name}' успешно создана. Количество документов: {final_count}")

    def get_count(self, collection_name: str) -> int:
        col = self.client.get_collection(name=collection_name)
        return col.count()

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Семантический поиск по вектору запроса с опциональной фильтрацией по метаданным."""
        col = self.client.get_collection(name=collection_name)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)
