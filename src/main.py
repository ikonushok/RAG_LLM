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

from tqdm import tqdm

from src.parser import build_index
from src.llm_integration import get_llm_answer
from telegram_bot import main as run_bot



def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    docs_dir = '../data/documents'
    index_dir = "../data/indexes"

    # 1/ Построение индексов - запустить в начале для сохранения индексов
    # brands = { "Daichi": "daichi.pdf", "Dantex": "dantex.pdf"}
    # for brand, filename in tqdm(brands.items(), desc="Building indexes.."):
    #     build_index(brand, os.path.join(docs_dir, filename), output_dir=index_dir)

    # 2/ Пример использования: загружаем нужный бренд
    # selected_brand = "Daichi"
    selected_brand = "Dantex"
    print(f'\nSelected_brand:\n {selected_brand}')

    with open(os.path.join(index_dir, f"text_{selected_brand}.pkl"), "rb") as f:
        texts = pickle.load(f)
    index = faiss.read_index(os.path.join(index_dir, f"index_{selected_brand}.faiss"))

    # Пример запроса
    # query = "Как очистить фильтр на Daichi ICE95AVQ1?"
    # query = "Как установить дренажный шланг на Daichi ICE95AVQ1?"
    query = "Как включить продув испарителя на сплите dantex RK-24SVGI?"
    print("Query:\n", query)
    response = get_llm_answer(query, texts, index)
    print("Response:\n", response)

    # 3/ (Опционально) Запуск Telegram-бота
    # run_bot()


if __name__ == "__main__":
    main()
