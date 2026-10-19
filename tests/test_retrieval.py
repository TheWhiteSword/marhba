"""Quick retrieval sanity check for the built index."""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex

INDEX_PATH = "data/index.tv"
META_PATH = "data/index_meta.json"
EMBED_MODEL = "hotchpotch/bekko-embedding-v1-a8m"

QUERIES = [
    "How much cash must I declare when entering Algeria?",
    "Can I bring a drone to Algeria?",
    "How long must my passport be valid?",
    "What is the duty-free allowance for tobacco?",
    "Can I bring my pet dog?",
    "Can I bring my prescription medication?",  # Should trigger "I don't know"
    "Quel est le seuil de déclaration de la monnaie?",  # French query
]


def main():
    model = SentenceTransformer(EMBED_MODEL)
    index = TurboQuantIndex.load(INDEX_PATH)

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    for q in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {q}")
        print('=' * 70)

        qvec = model.encode([q], normalize_embeddings=True).astype(np.float32)
        scores, slots = index.search(qvec, k=3)

        for rank, (slot, score) in enumerate(zip(slots[0], scores[0]), 1):
            m = meta[int(slot)]
            print(f"\n  [{rank}] score={score:.4f}  source={m['doc_id']}  lang={m['language']}")
            print(f"      heading: {m['heading_path']}")
            text = m['text']
            print(f"      text: {text[:200]}{'...' if len(text) > 200 else ''}")


if __name__ == "__main__":
    main()