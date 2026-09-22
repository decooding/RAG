"""
Скрипт подготовки и валидации датасетов:
1. Постатейный чанкинг (Structural): 1 чанк = 1 статья с метаданными.
2. Наивный чанкинг (Naive 500): окно 500 символов, шаг 425 символов (перекрытие 15%).
Сохраняет обработанные JSON в data/processed/ с выводом статистики.
"""

import json
from pathlib import Path
from typing import List
from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data.models import ArticleChunk, sanitize_metadata


def prepare_structural_dataset(input_json_path: Path) -> List[ArticleChunk]:
    """Загружает распарсенные статьи, санитизирует метаданные и формирует типизированные чанки."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    processed_chunks: List[ArticleChunk] = []
    for idx, item in enumerate(raw_chunks):
        meta = sanitize_metadata(item.get("metadata", {}))
        art_num = str(item.get("article_number", meta.get("article_number", "")))
        art_title = str(item.get("article_title", meta.get("article_title", "")))
        chunk_id = f"structural_{idx}_{art_num}"

        chunk = ArticleChunk(
            id=chunk_id,
            text=item.get("text", "").strip(),
            article_number=art_num,
            article_title=art_title,
            metadata=meta
        )
        processed_chunks.append(chunk)

    return processed_chunks


def prepare_naive_dataset(structural_chunks: List[ArticleChunk], window_size: int = 500, overlap_pct: float = 0.15) -> List[ArticleChunk]:
    """
    Разбивает текст статей наивным окном (500 символов с перекрытием 10-15%).
    Сохраняет связь с исходным номером статьи в метаданных.
    """
    step = int(window_size * (1.0 - overlap_pct))
    naive_chunks: List[ArticleChunk] = []

    global_idx = 0
    for art in structural_chunks:
        full_text = art.text
        if len(full_text) <= window_size:
            # Если статья короче окна, берем целиком
            n_chunk = ArticleChunk(
                id=f"naive_{global_idx}_{art.article_number}",
                text=full_text,
                article_number=art.article_number,
                article_title=art.article_title,
                metadata={
                    **art.metadata,
                    "chunk_type": "naive_500",
                    "naive_idx": global_idx
                }
            )
            naive_chunks.append(n_chunk)
            global_idx += 1
            continue

        start = 0
        while start < len(full_text):
            end = min(start + window_size, len(full_text))
            chunk_text = full_text[start:end].strip()
            if chunk_text:
                n_chunk = ArticleChunk(
                    id=f"naive_{global_idx}_{art.article_number}",
                    text=chunk_text,
                    article_number=art.article_number,
                    article_title=art.article_title,
                    metadata={
                        **art.metadata,
                        "chunk_type": "naive_500",
                        "naive_idx": global_idx,
                        "window_start": start,
                        "window_end": end
                    }
                )
                naive_chunks.append(n_chunk)
                global_idx += 1
            if end >= len(full_text):
                break
            start += step

    return naive_chunks


def print_stats(name: str, chunks: List[ArticleChunk]):
    """Выводит подробную статистику по датасету согласно DoD."""
    lens = [len(c.text) for c in chunks]
    words = [len(c.text.split()) for c in chunks]
    avg_len = sum(lens) / len(lens) if lens else 0
    avg_words = sum(words) / len(words) if words else 0

    print(f"=== Статистика датасета: {name} ===")
    print(f"  Всего чанков: {len(chunks)}")
    print(f"  Длина символов: мин={min(lens) if lens else 0}, макс={max(lens) if lens else 0}, среднее={avg_len:.1f}")
    print(f"  Количество слов: мин={min(words) if words else 0}, макс={max(words) if words else 0}, среднее={avg_words:.1f}")
    # Проверка отсутствия None в метаданных
    for c in chunks:
        for k, v in c.metadata.items():
            if v is None:
                raise ValueError(f"Обнаружен None в метаданных чанка {c.id}, ключ {k}")
    print("  [OK] Все метаданные строго валидированы и не содержат None.")
    print("-" * 50)


def main():
    source_json = PROJECT_ROOT / "src" / "data" / "1.json"
    if not source_json.exists():
        raise FileNotFoundError(f"Файл {source_json} не найден.")

    # 1. Постатейные чанки
    structural_chunks = prepare_structural_dataset(source_json)
    print_stats("Structural (Постатейный)", structural_chunks)
    out_structural = PROCESSED_DIR / "rk_articles_structural.json"
    with open(out_structural, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in structural_chunks], f, ensure_ascii=False, indent=2)
    print(f"Сохранено в: {out_structural}")

    # 2. Наивные чанки (500 символов)
    naive_500_chunks = prepare_naive_dataset(structural_chunks, window_size=500, overlap_pct=0.15)
    print_stats("Naive 500 (Окно 500 / перекрытие 15%)", naive_500_chunks)
    out_naive = PROCESSED_DIR / "rk_articles_naive_500.json"
    with open(out_naive, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in naive_500_chunks], f, ensure_ascii=False, indent=2)
    print(f"Сохранено в: {out_naive}")


if __name__ == "__main__":
    main()
