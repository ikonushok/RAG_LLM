import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from src.vectorizer import vectorize_text

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f'device: {device}')

# Загрузка локальной модели OpenHermes
local_path = "../local_models/local_openhermes"
tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True).to(device)

# Формирование промпта
def format_prompt(query: str, context: str) -> str:
    return f"[INST] Используя только следующий контекст, ответь на вопрос.\n\nКонтекст:\n{context}\n\nВопрос: {query} [/INST]"

# FAISS-поиск релевантных фрагментов
def get_relevant_text(query, faiss_index, original_texts, top_k=1):  # 2
    logging.debug(f"Query: {query}")
    query_vector = vectorize_text([query])
    logging.debug(f"Query vector: {query_vector}")

    query_vector = np.array(query_vector)
    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    distances, indices = faiss_index.search(query_vector, top_k)
    relevant_texts = [original_texts[i] for i in indices[0]]
    logging.debug(f"Relevant texts: {relevant_texts}")

    return relevant_texts

# Генерация ответа от LLM
# def get_llm_answer(query, original_texts, faiss_index):
#     relevant_texts = get_relevant_text(query, faiss_index, original_texts, top_k=2)
#     context = "\n---\n".join(relevant_texts)
#     prompt = format_prompt(query, context)
#
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,  # max_tokens=512,
#             do_sample=False,
#             # do_sample=True,
#             # temperature=0.7,
#             # top_p=0.9,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.replace(prompt, "").strip()

def get_llm_answer(query, original_texts, faiss_index):
    relevant_texts = get_relevant_text(query, faiss_index, original_texts, top_k=2)
    context = "\n---\n".join(relevant_texts)
    prompt = format_prompt(query, context)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()