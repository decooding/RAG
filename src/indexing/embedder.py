"""
Модуль генерации эмбеддингов:
1. LocalEmbedder: intfloat/multilingual-e5-small (через sentence-transformers, локально).
   Для E5 обязательно: префикс 'passage: ' для документов и 'query: ' для поисковых запросов.
2. GeminiEmbedder: text-embedding-004 (через официальный google-genai SDK).
   Включает экспоненциальный откат (backoff) при лимитах запросов (rate limits) и батчинг.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import time
import os
from src.config import LOCAL_EMBEDDING_MODEL_NAME, GEMINI_EMBEDDING_MODEL_NAME, GEMINI_API_KEY


class BaseEmbedder(ABC):
    """Абстрактный базовый класс для генераторов эмбеддингов."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Векторизация списка текстов документов."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Векторизация поискового запроса."""
        pass


class LocalEmbedder(BaseEmbedder):
    """Локальный эмбеддер на базе multilingual-e5-small."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL_NAME, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        # E5 требует префикс "passage: " для индексируемых фрагментов
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # E5 требует префикс "query: " для поисковых запросов
        prefixed = f"query: {text}"
        embedding = self.model.encode(
            prefixed,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding.tolist()


class GeminiEmbedder(BaseEmbedder):
    """Эмбеддер на базе Gemini API (text-embedding-004) с батчингом и retry."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_EMBEDDING_MODEL_NAME):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self._client = None

        if not self.api_key:
            # Предупреждение, если ключ не задан
            pass
        else:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        if not self.is_available:
            raise ValueError(
                "GEMINI_API_KEY не установлен. Укажите ключ в окружении или используйте LocalEmbedder."
            )

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        return res[0]

    def _embed_batch_with_retry(self, batch: List[str], max_retries: int = 5) -> List[List[float]]:
        """Отправка батча с экспоненциальным backoff при возникновении rate limit или сетевых ошибок."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Ошибка вызова Gemini Embeddings после {max_retries} попыток: {e}")
                time.sleep(delay)
                delay *= 2.0
        return []
