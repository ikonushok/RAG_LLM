import logging
from typing import List
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Инициализация модели
model = SentenceTransformer('all-MiniLM-L6-v2')

def filter_short_texts(texts, min_length=1):
    """Фильтрация слишком коротких текстов."""
    # Если тексты - это списки символов, объединяем их в строки
    texts = [''.join(text) if isinstance(text, list) else text for text in texts]
    # Фильтруем строки по длине
    return [text for text in texts if len(text) >= min_length]


def vectorize_text(texts: List[str]) -> List[List[float]]:
    """
    Принимает список осмысленных строк, фильтрует слишком короткие и отдает их эмбеддинги.
    """
    logging.debug(f"[vectorize_text] Input texts ({len(texts)}): {texts}")

    # 1) Фильтруем по длине (например, >=3 знака)
    filtered = filter_short_texts(texts)
    logging.info(f"[vectorize_text] After filter_short_texts: {filtered}")

    if not filtered:
        raise ValueError("Нет достаточных данных для векторизации.")

    # 2) Получаем эмбеддинги
    embeddings = model.encode(filtered)
    return embeddings.tolist()  # или просто embeddings, если не нужен список


def main():
    logging.basicConfig(level=logging.INFO)
    # Предположим, что original_texts уже извлечены
    original_texts = ['И', 'н', 'с', 'т', 'р', 'у', 'к', 'ц', 'и', 'я', ' ', 'п', 'о', '/n ', 'м', 'о', 'н', 'т', 'а',
                      'ж', 'у', ' ']
    logging.info(f"Original texts before vectorization: {original_texts}")
    try:
        text_vector = vectorize_text(original_texts)
        logging.debug(f"Vectorized texts: {text_vector}")
    except ValueError as e:
        logging.error(f"Error during vectorization: {e}")


if __name__ == "__main__":
    main()