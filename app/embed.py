"""Embedding model singleton — lazy load for Render."""

from sentence_transformers import SentenceTransformer

_model = None
MODEL_NAME = "hotchpotch/bekko-embedding-v1-a8m"


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)