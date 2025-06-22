import os
import warnings
# Подавляем все FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Подавляем конкретное предупреждение от PyTorch
warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pytree.*deprecated.*")
# Установка переменной окружения для устранения ошибок от MKL/OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from src.llm_integration import get_llm_answer

logging.basicConfig(level=logging.INFO)

BRAND, QUESTION = range(2)
BRANDS = ["Daichi", "Dantex"]
INDEX_DIR = "../data/indexes"
user_state = {}

async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [[brand.title()] for brand in BRANDS]
    await update.message.reply_text(
        "Про какой бренд вы хотите задать вопрос? 👇",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard,
            one_time_keyboard=True,
            resize_keyboard=True,
            input_field_placeholder="Выберите бренд"
        )
    )
    return BRAND

async def choose_brand(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    brand_input = update.message.text.strip().lower()
    brand = next((b for b in BRANDS if b.lower() == brand_input), None)

    if brand is None:
        await update.message.reply_text("Пожалуйста, выберите один из предложенных брендов.")
        return BRAND

    user_state[update.effective_chat.id] = {"brand": brand}

    await update.message.reply_text("Теперь задайте ваш вопрос.")
    return QUESTION

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    state = user_state.get(chat_id)

    if not state or "brand" not in state:
        await update.message.reply_text("Сначала выберите бренд, используя команду /start.")
        return ConversationHandler.END

    question = update.message.text
    brand = state["brand"]

    try:
        with open(os.path.join(INDEX_DIR, f"text_{brand}.pkl"), "rb") as f:
            texts = pickle.load(f)
        index = faiss.read_index(os.path.join(INDEX_DIR, f"index_{brand}.faiss"))

        answer = get_llm_answer(question, texts, index)
        await update.message.reply_text(answer)
    except Exception as e:
        logging.error(f"Ошибка при обработке вопроса: {e}")
        await update.message.reply_text("Произошла ошибка при обработке вашего запроса.")

    return ConversationHandler.END

def main():
    token = "7355008524:AAFEGyJvurpjGQCnXF9gly0_6aDY1gcRDao"

    app = ApplicationBuilder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_bot)],
        states={
            BRAND: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_brand)],
            QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_question)],
        },
        fallbacks=[]
    )

    app.add_handler(conv_handler)
    app.run_polling()

if __name__ == "__main__":
    main()