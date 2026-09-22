"""
Модуль для автоматической нарезки (chunking) текстов Кодексов Республики Казахстан
строго по статьям с сохранением номера статьи, названия и иерархических метаданных
(Раздел, Глава, Параграф) для RAG-систем.
"""

import re
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class ArticleChunk:
    """Представление чанка одной статьи Кодекса РК."""
    article_number: str
    article_title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование чанка в словарь."""
        data = asdict(self)
        # Объединяем основные атрибуты с метаданными для удобства RAG
        data["metadata"].update({
            "article_number": self.article_number,
            "article_title": self.article_title,
        })
        return data

    def to_langchain_document(self):
        """
        Преобразование в объект Document для LangChain / LlamaIndex:
        from langchain.schema import Document
        """
        try:
            from langchain.schema import Document
            meta = dict(self.metadata)
            meta["article_number"] = self.article_number
            meta["article_title"] = self.article_title
            return Document(page_content=self.text, metadata=meta)
        except ImportError:
            raise ImportError(
                "Пакет langchain не установлен. Используйте метод to_dict()."
            )


class RKCodeChunker:
    """
    Класс для разбиения текста законов и кодексов РК на чанки строго по статьям.
    Учитывает специфику законодательства РК (ИЗПИ / Әділет):
    - Номера с дефисами: 14-1, 270-4 и т.д.
    - Игнорирование сносок ('Сноска. Статья 5...') и примечаний ИЗПИ
    - Извлечение структурных разделов (Раздел, Глава, Параграф)
    - Флаг исключенных / утративших силу статей
    """

    # Регулярное выражение для поиска заголовка статьи:
    # 1. ^[ \t]* - начало строки с возможными пробелами/табуляцией
    # 2. (?:СТАТЬЯ|Статья) - ключевое слово (с учетом регистра)
    # 3. (?:\s+|\n) - пробел(ы) или перенос строки перед номером
    # 4. (?P<num>[0-9]+(?:-[0-9]+)*[а-яА-Яa-zA-Z]?) - номер статьи (1, 14-1, 270-4 и т.д.)
    # 5. Разделитель и название статьи:
    #    - точка с пробелом/переносом: \.(?!\d)(?:[ \t]*(?P<title1>[^\n\r]*)|$)
    #    - тире/двоеточие: [ \t]*[:–—][ \t]*(?P<title2>[^\n\r]*)
    #    - конец строки (когда названия нет, как в Конституции): [ \t]*$
    ARTICLE_PATTERN = re.compile(
        r'(?m)^[ \t]*(?:СТАТЬЯ|Статья)\s+'
        r'(?P<num>[0-9]+(?:-[0-9]+)*[а-яА-Яa-zA-Z]?)'
        r'(?:'
        r'\.(?!\d)(?:[ \t]*(?P<title1>[^\n\r]*)|$)'
        r'|[ \t]*[:–—][ \t]*(?P<title2>[^\n\r]*)'
        r'|[ \t]*$'
        r')'
    )

    # Паттерны для отслеживания структурного контекста (Разделы, Главы, Параграфы)
    SECTION_PATTERN = re.compile(
        r'(?m)^[ \t]*(?:РАЗДЕЛ|Раздел)\s+([IVXLCDM0-9]+(?:-[0-9]+)?\.?[ \t]*[^\n\r]*)',
        re.IGNORECASE
    )
    CHAPTER_PATTERN = re.compile(
        r'(?m)^[ \t]*(?:ГЛАВА|Глава)\s+([0-9]+(?:-[0-9]+)?\.?[ \t]*[^\n\r]*)',
        re.IGNORECASE
    )
    PARAGRAPH_PATTERN = re.compile(
        r'(?m)^[ \t]*(?:ПАРАГРАФ|Параграф|§)\s+([0-9]+(?:-[0-9]+)?\.?[ \t]*[^\n\r]*)',
        re.IGNORECASE
    )

    # Паттерн для проверки, исключена ли статья
    REPEALED_PATTERN = re.compile(
        r'(?:исключен[аоы]?|утратил[аоы]?\s+силу)',
        re.IGNORECASE
    )

    def __init__(self, include_preamble: bool = True, clean_whitespace: bool = True):
        """
        :param include_preamble: Если True, текст до первой статьи (преамбула, дата принятия,
                                 общие изменения) будет сохранен как отдельный чанк с номером 'ПРЕАМБУЛА'.
        :param clean_whitespace: Очищать ли избыточные пробелы и пустые строки в тексте чанка.
        """
        self.include_preamble = include_preamble
        self.clean_whitespace = clean_whitespace

    def _clean_text(self, text: str) -> str:
        """Нормализация пробелов и переносов строк."""
        if not self.clean_whitespace:
            return text.strip()
        # Замена последовательностей из 3+ пустых строк на 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Убираем пробелы в конце строк
        lines = [line.rstrip() for line in text.splitlines()]
        return '\n'.join(lines).strip()

    def _find_structure_positions(self, text: str) -> List[Dict[str, Any]]:
        """Находит позиции всех Разделов, Глав и Параграфов для точной привязки к статьям."""
        structures = []
        for match in self.SECTION_PATTERN.finditer(text):
            structures.append({
                "type": "section",
                "start": match.start(),
                "title": f"Раздел {match.group(1).strip()}"
            })
        for match in self.CHAPTER_PATTERN.finditer(text):
            structures.append({
                "type": "chapter",
                "start": match.start(),
                "title": f"Глава {match.group(1).strip()}"
            })
        for match in self.PARAGRAPH_PATTERN.finditer(text):
            structures.append({
                "type": "paragraph",
                "start": match.start(),
                "title": f"Параграф {match.group(1).strip()}"
            })

        # Сортируем по позиции в тексте
        structures.sort(key=lambda x: x["start"])
        return structures

    def chunk_text(self, text: str, source: Optional[str] = None) -> List[ArticleChunk]:
        """
        Разбивает переданный текст кодекса строго по статьям.

        :param text: Исходный текст кодекса / закона РК.
        :param source: Имя источника (файл, URL, название кодекса) для метаданных.
        :return: Список объектов ArticleChunk.
        """
        if not text or not text.strip():
            return []

        # 1. Поиск всех заголовков статей
        article_matches = list(self.ARTICLE_PATTERN.finditer(text))
        if not article_matches:
            # Статьи не найдены
            return []

        # 2. Поиск структурных заголовков (Раздел, Глава, Параграф)
        structures = self._find_structure_positions(text)

        chunks: List[ArticleChunk] = []
        current_section = ""
        current_chapter = ""
        current_paragraph = ""
        struct_idx = 0

        # Вспомогательная функция для обновления текущей структуры до заданной позиции
        def update_structure(pos: int):
            nonlocal current_section, current_chapter, current_paragraph, struct_idx
            while struct_idx < len(structures) and structures[struct_idx]["start"] <= pos:
                st = structures[struct_idx]
                if st["type"] == "section":
                    current_section = st["title"]
                elif st["type"] == "chapter":
                    current_chapter = st["title"]
                elif st["type"] == "paragraph":
                    current_paragraph = st["title"]
                struct_idx += 1

        # 3. Обработка преамбулы (текст до первой статьи)
        first_art_start = article_matches[0].start()
        if self.include_preamble and first_art_start > 0:
            preamble_raw = text[:first_art_start].strip()
            # Обновляем структуру до первой статьи
            update_structure(first_art_start)
            if len(preamble_raw) > 30:  # Игнорируем совсем короткие префиксы
                preamble_text = self._clean_text(preamble_raw)
                first_line = preamble_text.splitlines()[0] if preamble_text else ""
                chunks.append(ArticleChunk(
                    article_number="ПРЕАМБУЛА",
                    article_title=first_line[:150],
                    text=preamble_text,
                    metadata={
                        "source": source,
                        "type": "preamble",
                        "section": current_section or None,
                        "chapter": current_chapter or None,
                        "paragraph": current_paragraph or None,
                        "is_repealed": False,
                        "chunk_index": 0,
                        "char_count": len(preamble_text),
                        "word_count": len(preamble_text.split()),
                    }
                ))

        # 4. Нарезка по статьям строго от начала текущей до начала следующей
        for i, match in enumerate(article_matches):
            art_start = match.start()
            art_end = article_matches[i + 1].start() if (i + 1 < len(article_matches)) else len(text)

            # Обновляем текущую структуру (Раздел / Глава / Параграф)
            update_structure(art_start)

            # Извлекаем номер статьи
            art_num = match.group("num")

            # Извлекаем название статьи
            title = (match.group("title1") or match.group("title2") or "").strip()

            # Текст статьи
            article_raw = text[art_start:art_end]
            article_clean = self._clean_text(article_raw)

            # Если название не было в той же строке (например, перенос на следующую строку),
            # извлекаем первую непустую строку после заголовка
            if not title:
                lines = [ln.strip() for ln in article_clean.splitlines() if ln.strip()]
                if len(lines) > 1 and not re.match(r'^(?:[0-9]+\.|\([0-9]+\)|[0-9]+\))', lines[1]):
                    # Если вторая строка не начинается с номера пункта (1., 1), (1)),
                    # то это, вероятно, заголовок статьи
                    title = lines[1]
                else:
                    title = f"Статья {art_num}"

            # Проверка на статус исключенной статьи
            is_repealed = bool(self.REPEALED_PATTERN.search(title)) or (
                len(article_clean) < 300 and bool(self.REPEALED_PATTERN.search(article_clean))
            )

            chunk_idx = len(chunks)
            chunk = ArticleChunk(
                article_number=art_num,
                article_title=title,
                text=article_clean,
                metadata={
                    "source": source,
                    "type": "article",
                    "section": current_section or None,
                    "chapter": current_chapter or None,
                    "paragraph": current_paragraph or None,
                    "is_repealed": is_repealed,
                    "chunk_index": chunk_idx,
                    "char_count": len(article_clean),
                    "word_count": len(article_clean.split()),
                }
            )
            chunks.append(chunk)

        return chunks

    def chunk_pdf(self, pdf_path: Union[str, Path]) -> List[ArticleChunk]:
        """
        Извлекает текст из PDF документа (через PyMuPDF/fitz) и разбивает на чанки по статьям.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "Для работы с PDF установите PyMuPDF: pip install pymupdf"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        full_text = "\n".join(pages_text)

        return self.chunk_text(full_text, source=pdf_path.name)

    def chunk_file(self, file_path: Union[str, Path], encoding: str = "utf-8") -> List[ArticleChunk]:
        """Универсальная нарезка текстового (.txt) или PDF (.pdf) файла."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self.chunk_pdf(path)
        else:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                text = f.read()
            return self.chunk_text(text, source=path.name)


def main():
    """CLI интерфейс для быстрой нарезки документов."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Автоматическая нарезка Кодексов РК на чанки строго по статьям."
    )
    parser.add_argument("input_path", help="Путь к текстовому файлу (.txt) или PDF (.pdf)")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата (JSON). Если не указан, выводится статистика.")
    parser.add_argument("--no-preamble", action="store_true", help="Не включать преамбулу как отдельный чанк")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество выводимых статей для предпросмотра")

    args = parser.parse_args()

    chunker = RKCodeChunker(include_preamble=not args.no_preamble)
    print(f"Обработка файла: {args.input_path}...")
    chunks = chunker.chunk_file(args.input_path)

    print(f"Успешно извлечено {len(chunks)} чанков (статей)!")

    # Статистика
    repealed_count = sum(1 for c in chunks if c.metadata.get("is_repealed"))
    print(f"- Обычных статей: {len(chunks) - repealed_count}")
    print(f"- Исключенных / утративших силу: {repealed_count}")

    # Предпросмотр первых нескольких чанков
    preview_limit = args.limit or min(3, len(chunks))
    print(f"\n--- Предпросмотр первых {preview_limit} чанков: ---")
    for i in range(preview_limit):
        c = chunks[i]
        print(f"[{i+1}] Статья: {c.article_number} | Заголовок: {c.article_title}")
        print(f"    Глава: {c.metadata.get('chapter')} | Раздел: {c.metadata.get('section')}")
        print(f"    Символов: {c.metadata.get('char_count')} | Слов: {c.metadata.get('word_count')}")
        print(f"    Начало текста: {c.text[:120].replace(chr(10), ' ')}...")
        print("-" * 50)

    # Сохранение в JSON
    if args.output:
        out_path = Path(args.output)
        data = [c.to_dict() for c in chunks]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в файл: {out_path.resolve()}")


if __name__ == "__main__":
    main()
