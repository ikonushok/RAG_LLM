from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Используем MiniLM или аналог для векторизации
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def filter_short_texts(texts, min_length=1):
    """Фильтрация слишком коротких текстов."""
    # Если тексты - это списки символов, объединяем их в строки
    texts = [''.join(text) if isinstance(text, list) else text for text in texts]
    # Фильтруем строки по длине
    return [text for text in texts if len(text) >= min_length]

@torch.no_grad()
def vectorize_text(texts):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Используем [CLS] токен
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()