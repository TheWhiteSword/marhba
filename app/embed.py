"""Embedding model singleton — ONNX backend for low memory on Render."""
"""Torch-free ONNX embedding — reads from committed LFS model."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_model = None
_tokenizer = None

CACHE_BASE = Path("data/model_cache/blobs")


def _find_blob(pattern: str) -> Path:
    matches = list(CACHE_BASE.glob(pattern))
    if not matches:
        raise RuntimeError(f"No blob matching {pattern} under {CACHE_BASE}")
    return matches[0]


def get_model():
    global _model, _tokenizer
    if _model is None:
        # ONNX blob is 125 MB in blobs/f7/
        # Tokenizer blob is 33 MB in blobs/33/
        onnx_file = _find_blob("f7/*")
        tokenizer_file = _find_blob("33/*")

        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.intra_op_num_threads = 1

        _model = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _tokenizer = Tokenizer.from_file(str(tokenizer_file))
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

    last_hidden = outputs[0]
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    embeddings = summed / np.maximum(counts, 1e-9)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings.astype(np.float32)