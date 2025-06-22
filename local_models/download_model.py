# from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Укажите путь для сохранения модели
# local_model_path = "./local_mistral"  # Папка, в которую будет сохранена модель
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
local_model_path = "./local_openhermes"
model_name = "teknium/OpenHermes-2.5-Mistral-7B"


# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# Сохранение локально
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print(f"Модель и токенизатор сохранены в {local_model_path}")



