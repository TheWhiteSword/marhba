"""Embedding model singleton — ONNX backend for low memory on Render."""
"""Torch-free ONNX embedding for Marhba."""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

_model = None
_tokenizer = None

MODEL_ID = "hotchpotch/bekko-embedding-v1-a8m"
ONNX_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"


def get_model():
    global _model, _tokenizer
    if _model is None:
        onnx_path = hf_hub_download(repo_id=MODEL_ID, filename=ONNX_FILE)
        tokenizer_path = hf_hub_download(repo_id=MODEL_ID, filename=TOKENIZER_FILE)

        _model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        _tokenizer.enable_truncation(max_length=512)
        _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    return _model, _tokenizer


def embed(texts: list[str]) -> np.ndarray:
    model, tokenizer = get_model()

    encoded = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    outputs = model.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })

    # Mean pooling over token embeddings, respecting attention mask
    last_hidden = outputs[0]  # shape: (batch, seq_len, hidden_dim)
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    embeddings = summed / np.maximum(counts, 1e-9)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    return embeddings.astype(np.float32)