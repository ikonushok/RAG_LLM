import pdfplumber
from PyPDF2 import PdfReader
from src.utils import timer_decorator


# @timer_decorator
def extract_text_from_pdf(pdf_path, encoding='utf-8'):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Ошибка при парсинге документа: {e}")
        return exit()


# @timer_decorator
# def extract_text_from_pdf(file_path):
#     pdf_reader = PdfReader(file_path)
#     text = ''
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text