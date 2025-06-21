import faiss
import numpy as np

# Маппинг текстов на индексы
text_to_index_mapping = {}


def filter_short_texts(texts, min_length=3):
    """Фильтрация слишком коротких текстов."""
    # Если тексты - это списки символов, объединяем их в строки
    texts = [''.join(text) if isinstance(text, list) else text for text in texts]
    # Фильтруем строки по длине
    return [text for text in texts if len(text) >= min_length]

def create_faiss_index(vectors, texts):
    # Фильтруем короткие тексты
    filtered_texts = filter_short_texts(texts)

    # Если после фильтрации нет текста, выбрасываем исключение
    if not filtered_texts:
        raise ValueError("No valid texts available after filtering.")

    # Создание FAISS индекса
    if len(vectors.shape) == 1:
        vectors = np.expand_dims(vectors, axis=0)

    index = faiss.IndexFlatL2(vectors.shape[1])  # FAISS индекс для поиска по L2 расстоянию
    index.add(np.array(vectors, dtype=np.float32))  # Добавляем векторы в индекс

    # Маппинг текстов на индексы
    for i, text in enumerate(filtered_texts):
        text_to_index_mapping[i] = text

    return index