import faiss
import numpy as np

from src.vectorizer import filter_short_texts

# Маппинг текстов на индексы
text_to_index_mapping = {}



def create_faiss_index(vectors, texts):
    # Фильтруем короткие тексты
    filtered_texts = filter_short_texts(texts)

    # Если после фильтрации нет текста, выбрасываем исключение
    if not filtered_texts:
        raise ValueError("No valid texts available after filtering.")

    # Преобразуем векторы в массив NumPy (если это еще не так)
    vectors = np.array(vectors)

    # Если векторы одномерные (один текст), добавляем еще одну ось
    if len(vectors.shape) == 1:
        vectors = np.expand_dims(vectors, axis=0)

    # Создание FAISS индекса
    index = faiss.IndexFlatL2(vectors.shape[1])  # FAISS индекс для поиска по L2 расстоянию
    index.add(np.array(vectors, dtype=np.float32))  # Добавляем векторы в индекс

    # Маппинг текстов на индексы
    for i, text in enumerate(filtered_texts):
        text_to_index_mapping[i] = text

    return index
