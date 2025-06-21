import os

from src.llm_integration import get_llm_answer
from src.parser import extract_text_from_pdf
from src.vectorizer import vectorize_text
from src.faiss_index import create_faiss_index
# from src.telegram_bot import start_bot



def main():
    # Пример парсинга документа
    document_path = '../data'
    documents_names = os.listdir(document_path)
    if '.DS_Store' in documents_names:
        documents_names.remove('.DS_Store')
    # print(documents_names)

    # Парсинг и векторизация
    original_texts = extract_text_from_pdf(f'{document_path}/{documents_names[1]}')
    print(f'Document: {documents_names[1]} is parsed')
    text_vector = vectorize_text(original_texts)
    print(f"Vectors obtained: {text_vector.shape}")

    # Создание и поиск в FAISS
    faiss_index = create_faiss_index(text_vector, original_texts)
    print("faiss_index created")

    # Checking faiss working
    query = "Как установить дренажный шланг на Daichi ICE95AVQ1?"
    response = get_llm_answer(query, original_texts, faiss_index)
    print(f"Response: {response}")

    # Запуск Telegram-бота
    # start_bot(faiss_index)  # Передаем индекс в бота для использования


if __name__ == "__main__":
    main()
