import os
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from src.utils import timer_decorator
from src.vectorizer import vectorize_text

# Загрузка модели BERT (Base) и токенизатора
# Путь к локальной папке, где хранится модель и токенизатор
local_model_path = "../local_models/local_distilbert"
print(f'Model path exists: {os.path.exists(local_model_path)}')  # Проверка существования папки
print(f'Files in model path: {os.listdir(local_model_path)}')     # Содержимое папки
model = DistilBertForQuestionAnswering.from_pretrained(local_model_path, local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained(local_model_path, local_files_only=True)


# Функция для получения наиболее релевантных фрагментов текста с использованием FAISS
def get_relevant_text(query, faiss_index, original_texts, top_k=1):
    # Проверка на пустой запрос
    if not query.strip():
        raise ValueError("Запрос не может быть пустым.")

    # Преобразуем запрос в вектор
    query_vector = vectorize_text([query])  # Векторизуем запрос

    # Поиск наиболее похожих документов
    distances, indices = faiss_index.search(query_vector, top_k)

    # Извлечение текста по найденным индексам
    relevant_texts = [original_texts[i] for i in indices[0]]

    print(f"Relevant texts found: {relevant_texts}")  # Выводим релевантные фрагменты
    return relevant_texts


# Модифицированная функция get_llm_answer, которая теперь использует FAISS для поиска контекста
def get_llm_answer(query, original_texts, faiss_index):
    # Шаг 1: Получаем наиболее релевантные фрагменты с использованием FAISS
    relevant_texts = get_relevant_text(query, faiss_index, original_texts)

    # Проверим, что релевантные фрагменты не пустые
    if not relevant_texts:
        print("Не найдены релевантные фрагменты.")
        return "Ответ не найден."

    # Шаг 2: Передаем найденный текст в модель для ответа
    context = " ".join(relevant_texts)  # Объединяем все фрагменты в один контекст
    print(f"Context for LLM: {context}")  # Выводим контекст для проверки

    inputs = tokenizer(query, context, return_tensors="pt", truncation='only_second', padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Извлекаем начало и конец ответа
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    # Проверка на корректность начала и конца ответа
    if answer_end < answer_start:
        print("Ошибка в извлечении ответа. Проверка токенов.")
        return "Ответ не найден."

    answer = context[answer_start:answer_end + 1]
    return answer







