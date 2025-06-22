import os
import re
import pickle
import logging

from typing import List

import faiss
from pdfminer.high_level import extract_text

from src.vectorizer import vectorize_text
from src.faiss_index import create_faiss_index



def extract_text_from_pdf(pdf_path):
    raw = extract_text(pdf_path) or ""
    return raw

def split_into_paragraphs(raw: str) -> List[str]:
    """
    Разбивает сырой текст на параграфы по пустым строкам.
    """
    lines = raw.splitlines()
    paras, buf = [], []
    for line in lines:
        if line.strip() == "":
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        paras.append(" ".join(buf))
    return paras

def preprocess_paragraph(p: str) -> str:
    """
    1. Склеивает дефисный перенос: "мон-\nтаж" → "монтаж"
    2. Заменяет остатки переносов строк на пробелы.
    3. Сводит подряд идущие пробелы к одному.
    4. Убирает пробелы по краям.
    """
    # 1) дефисы-переносы
    p = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', p)
    # 2) остальные переводы строк
    p = p.replace("\n", " ")
    # 3) множественные пробелы → один
    p = re.sub(r'\s+', " ", p)
    # 4) обрезка
    return p.strip()

def parse_pdf(pdf_path: str, min_length: int = 30) -> List[str]:
    """
    Возвращает список «чистых» параграфов длиной >= min_length.
    """
    raw = extract_text_from_pdf(pdf_path)
    if not raw:
        return []

    paras = split_into_paragraphs(raw)
    cleaned = [preprocess_paragraph(p) for p in paras]
    # Оставляем только «достаточно длинные» фрагменты
    return [p for p in cleaned if len(p) >= min_length]

def build_index(brand_name, pdf_path, output_dir="../data/indexes"):

    os.makedirs(output_dir, exist_ok=True)
    text = parse_pdf(pdf_path)

    if not text:
        print(f"[{brand_name}] PDF пуст или не распарсен.")
        return

    vectors = vectorize_text(text)
    index = create_faiss_index(vectors, text)

    with open(os.path.join(output_dir, f"text_{brand_name}.pkl"), "wb") as f:
        pickle.dump(text, f)

    faiss.write_index(index, os.path.join(output_dir, f"index_{brand_name}.faiss"))
    logging.info(f"{brand_name}:\tИндекс и текст сохранены.")


# Пример использования:
def main():
    logging.basicConfig(level=logging.INFO)

    # 1) Парсим PDF и получаем абзацы
    paras = parse_pdf("../data/documents/sample_2.pdf")
    logging.info(f"Parsed {len(paras)} paragraphs.")

    if not paras:
        logging.error("Нечего векторизовать — парсер вернул пустой список.")
        return

    # 2) Векторизуем
    try:
        vectors = vectorize_text(paras)
        logging.info(f"Got {len(vectors)} векторных представлений.")
    except Exception as e:
        logging.error(f"Ошибка при векторизации: {e}")

if __name__ == "__main__":
    main()
