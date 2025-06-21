
from pdfminer.high_level import extract_text
import logging


def extract_text_from_pdf(pdf_path):
    """
    Извлекает текст из PDF файла с помощью pdfminer.six
    """
    try:
        # Извлекаем текст из PDF
        text = extract_text(pdf_path)
        if not text:
            logging.warning(f"Текст из файла {pdf_path} не был извлечен.")
        return text
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из PDF: {e}")
        return ""


def preprocess_text(text):
    """
    Преобразует извлеченный текст: удаляет лишние пробелы и символы новой строки
    """
    # Убираем лишние пробелы, символы новой строки и табуляции
    processed_text = ' '.join(text.split())
    return processed_text


def parse_pdf(pdf_path):
    """
    Основная функция для извлечения и обработки текста из PDF
    """
    # Извлекаем текст из PDF
    text = extract_text_from_pdf(pdf_path)

    # Если текст извлечен, то обрабатываем его
    if text:
        processed_text = preprocess_text(text)
        return processed_text
    else:
        return None


# Пример использования:
if __name__ == '__main__':
    pdf_path = '../data/sample_2.pdf'  # Путь к вашему PDF файлу
    text = parse_pdf(pdf_path)

    if text:
        print("Извлеченный и обработанный текст:")
        print(text)
    else:
        print("Не удалось извлечь текст из файла.")
