from sentence_transformers import SentenceTransformer

from src.utils import timer_decorator


# @timer_decorator
def vectorize_text(text):
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # model = SentenceTransformer('bert-base-nli-mean-tokens')
    vectors = model.encode([text])
    return vectors

