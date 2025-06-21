import warnings
from src.llm_integration import get_llm_answer

warnings.filterwarnings("ignore", category=FutureWarning)

# Простейший пример с вручную заданными текстами (эмуляция FAISS)
def dummy_faiss_index(*args, **kwargs):
    class DummyIndex:
        def search(self, query_vector, top_k):
            return None, [[0]]
    return DummyIndex()

# Тестовые данные
query = "Как установить дренажный шланг на Daichi ICE95AVQ1?"
original_texts = ["""
Шаг 6. Установка дренажного шланга
1. Присоедините дренажный шланг к выходной трубе внутреннего блока.
2. Оберните место соединения лентой.

Примечание: Для предотвращения конденсации влаги шланг нужно теплоизолировать.
"""]

# Используем поддельный FAISS индекс (он всегда возвращает первый текст)
faiss_index = dummy_faiss_index()

# Получаем ответ от модели
answer = get_llm_answer(query, original_texts, faiss_index)
print("\nОтвет от OpenHermes:\n")
print(answer)
