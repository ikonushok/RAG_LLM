import logging
import os
# import logging

from src.llm_integration import get_llm_answer
from src.parser import parse_pdf
from src.vectorizer import vectorize_text
from src.faiss_index import create_faiss_index
# from src.telegram_bot import start_bot

def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # 1) Собираем список PDF файлов в папке documents/documents
    docs_dir = '../documents'
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {docs_dir}")
        return

    # Выбираем первый документ (или любой другой по индексу)
    pdf_name = pdf_files[1]
    pdf_path = os.path.join(docs_dir, pdf_name)
    logging.info(f"Parsing document: {pdf_name}")

    # 2) Парсим PDF → получаем список «чистых» параграфов
    original_texts = parse_pdf(pdf_path)
    logging.info(f"Extracted {len(original_texts)} paragraphs")

    if not original_texts:
        print("Parser returned no text to vectorize")
        return

    # 3) Векторизуем
    text_vectors = vectorize_text(original_texts)
    # Если это список списков, измеряем так:
    num_vecs = len(text_vectors)
    dim = len(text_vectors[0]) if num_vecs > 0 else 0
    logging.info(f"Obtained {num_vecs} vectors of dimension {dim}")

    # 4) Создаём FAISS-индекс
    faiss_index = create_faiss_index(text_vectors, original_texts)
    logging.info("FAISS index created")

    # 5) Пробный запрос к LLM через RAG
    query = "Как очистить фильтр на Daichi ICE95AVQ1?"
    # query = "Как установить дренажный шланг на Daichi ICE95AVQ1?"
    print("\nQuery:\n", query)
    response = get_llm_answer(query, original_texts, faiss_index)
    print("\nResponse:\n", response)

    # 6) (Опционально) Запуск Telegram-бота
    # start_bot(faiss_index)


if __name__ == "__main__":
    main()
