"""Torch-free ONNX embedding — downloads from HF during build."""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

_model = None
_tokenizer = None

MODEL_ID = "hotchpotch/bekko-embedding-v1-a8m"
ONNX_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"

CACHE_DIR = Path("data/model_cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_model():
    global _model, _tokenizer
    if _model is None:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

        onnx_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=ONNX_FILE,
            cache_dir=str(CACHE_DIR),
        )
        tokenizer_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=TOKENIZER_FILE,
            cache_dir=str(CACHE_DIR),
        )

        # Defensive check: both paths must be real files, not .refs/.lock
        for label, p in (("ONNX", onnx_path), ("Tokenizer", tokenizer_path)):
            path_obj = Path(p)
            if not path_obj.exists():
                raise RuntimeError(f"{label} file missing: {p}")
            if path_obj.suffix in (".refs", ".lock"):
                raise RuntimeError(
                    f"{label} path points to metadata file, not model: {p}"
                )

        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.intra_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        _model = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
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

    last_hidden = outputs[0]
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    embeddings = summed / np.maximum(counts, 1e-9)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings.astype(np.float32)