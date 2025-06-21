import os
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer



# Загрузка модели и токенизатора
local_model_path = "./local_distilbert"
# Проверяем, существует ли папка и выводим содержимое
print(f'Model path exists: {os.path.exists(local_model_path)}')  # Проверяет, существует ли папка
print(f'Files in model path: {os.listdir(local_model_path)}')     # Показывает содержимое папки
model = DistilBertForQuestionAnswering.from_pretrained(local_model_path, local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained(local_model_path, local_files_only=True)

# Функция для получения ответа на вопрос
def get_llm_answer(query, context):
    try:
        # Токенизация входного текста
        inputs = tokenizer(query, context, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Получение предсказания от модели
        with torch.no_grad():  # Отключаем вычисление градиентов, так как нам не нужно их в этом случае
            outputs = model(**inputs)

        # Для задачи вопрос-ответ извлекаем результат
        answer_start = torch.argmax(outputs.start_logits)  # Индекс начала ответа
        answer_end = torch.argmax(outputs.end_logits)  # Индекс конца ответа
        answer = context[answer_start:answer_end+1]  # Ответ - от начала до конца

    except Exception as e:
        # Обработка ошибок
        answer = f"Произошла ошибка: {str(e)}"

    return answer


# Пример запроса
query = "Столица Франции?"
context = "Столица Франции - Париж."
response = get_llm_answer(query, context)
print(f'\nquery: {query}\ncontext: {context}\nresponse: {response}')





