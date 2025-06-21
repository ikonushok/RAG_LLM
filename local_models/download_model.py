from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

# Укажите путь для сохранения модели
local_model_path = "./local_distilbert"  # Папка, в которую будет сохранена модель

# Загрузка и сохранение модели и токенизатора
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Сохраняем локально
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)

print(f"Модель и токенизатор сохранены в {local_model_path}")
