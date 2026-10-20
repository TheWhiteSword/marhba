"""Cloud embedding via Google Gemini API — no local model, fits Render free tier."""

import os
import time

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-embedding-001"
EMBED_DIM = 768
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:embedContent"

MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds


def embed(texts: list[str]) -> np.ndarray:
    """Get embeddings from Google Gemini API with retry on 429."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey"
        )

    embeddings_list = []
    with httpx.Client(timeout=60.0) as client:
        for text in texts:
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.post(
                        API_URL,
                        headers={"x-goog-api-key": GEMINI_API_KEY},
                        json={
                            "model": f"models/{MODEL}",
                            "content": {"parts": [{"text": text}]},
                            "output_dimensionality": EMBED_DIM,
                        },
                    )
                    if response.status_code == 429:
                        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                        delay = BASE_DELAY * (2 ** attempt)
                        print(f"      429 rate limit, waiting {delay:.0f}s...")
                        time.sleep(delay)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    embeddings_list.append(data["embedding"]["values"])
                    break  # Success, move to next text
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                        continue  # Will be caught by the retry loop
                    raise
            else:
                raise RuntimeError(
                    f"Failed to embed text after {MAX_RETRIES} retries due to rate limits. "
                    "Wait a few minutes and run the build again."
                )

    embeddings = np.array(embeddings_list, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings