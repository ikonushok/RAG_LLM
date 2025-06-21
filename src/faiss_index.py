import faiss
import numpy as np

from src.utils import timer_decorator

texts = []  # Здесь будут храниться оригинальные тексты

@timer_decorator
def create_faiss_index(vectors, original_texts):
    global texts
    texts = original_texts  # Сохраняем оригинальные тексты для последующего поиска

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

# @timer_decorator
@timer_decorator
def search_faiss(query_vector, index, k=1):
    D, I = index.search(np.array(query_vector), k)
    return D, I, texts

