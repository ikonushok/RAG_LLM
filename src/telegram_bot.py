import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from src.faiss_index import create_faiss_index
from src.llm_integration import get_llm_answer
from src.vectorizer import vectorize_text  # Импортируем функцию для векторизации

# Переменная для хранения индекса FAISS
faiss_index = None

# Функция стартовой команды
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Задай свой вопрос.")


# Функция для обработки сообщений
async def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text

    # Векторизация текста пользователя
    query_vector = vectorize_text(user_message)  # Преобразуем текст в вектор

    # Проверка формы вектора
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)  # Преобразуем вектор в нужную форму (1, d)

    # Ищем ответ в FAISS
    D, I, texts = create_faiss_index(query_vector, faiss_index, k=1)  # Передаем индекс для поиска
    if I is not None and len(I) > 0:
        # Извлекаем текст, соответствующий найденному индексу
        result_text = texts[I[0][0]]  # Получаем текст по индексу
        await update.message.reply_text(f"Результат поиска в FAISS {D[0]}: {result_text}")
    else:
        # Если не нашли, используем LLM
        llm_answer = get_llm_answer(user_message)
        await update.message.reply_text(llm_answer)


# Функция для запуска бота
def start_bot(index):
    global faiss_index
    faiss_index = index  # Передаем индекс FAISS в глобальную переменную

    # Инициализация приложения и передача API ключа
    application = Application.builder().token("7355008524:AAFEGyJvurpjGQCnXF9gly0_6aDY1gcRDao").build()

    # Добавление обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    application.run_polling()


if __name__ == "__main__":
    # t.me/ZeBrains_test_bot
    # Use this token to access the HTTP API: 7355008524:AAFEGyJvurpjGQCnXF9gly0_6aDY1gcRDao
    start_bot(faiss_index)  # Передаем индекс FAISS при запуске бота

