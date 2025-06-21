# Пример кода для взаимодействия с Telegram API
import logging
from telegram import Update
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Здесь мы применяем новый код фильтрации
from faiss_index import create_faiss_index
from vectorizer import vectorize_text

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Привет! Задай вопрос по инструкции.")

def handle_message(update: Update, context: CallbackContext) -> None:
    user_query = update.message.text
    relevant_texts = get_relevant_text(user_query)  # Функция поиска релевантных текстов

    # Отправляем пользователю найденные данные
    update.message.reply_text(f"Ответ: {relevant_texts}")

def get_relevant_text(query):
    """Функция для получения релевантных текстов"""
    query_vector = vectorize_text([query])
    faiss_index = create_faiss_index(query_vector, original_texts)
    # Функция для получения релевантных текстов из FAISS индекса
    return faiss_index.search(query_vector, k=5)

def main() -> None:
    """Запуск бота"""
    updater = Updater("YOUR_TOKEN")

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()