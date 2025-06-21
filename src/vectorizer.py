import logging

from sentence_transformers import SentenceTransformer

from src.faiss_index import filter_short_texts

# Инициализация модели
model = SentenceTransformer('all-MiniLM-L6-v2')  # или другая модель


def vectorize_text(texts):
    # Преобразуем символы в строки, если необходимо
    texts = [''.join(text) if isinstance(text, list) else text for text in texts]
    print(f"Texts to vectorize: {texts}")

    # Фильтруем короткие тексты
    filtered_texts = filter_short_texts(texts)

    # Логируем данные, которые остались после фильтрации
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Filtered texts: {filtered_texts}")

    if not filtered_texts:
        raise ValueError("All texts are too short to process.")

    # Векторизация текста
    vectors = model.encode(filtered_texts)
    return vectors