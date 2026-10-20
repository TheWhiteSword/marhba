"""RAG retrieval and generation for Marhba."""

import json
import os
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from turbovec import TurboQuantIndex

from app.embed import embed

load_dotenv()

INDEX_PATH = Path("data/index.tv")
META_PATH = Path("data/index_meta.json")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-super-vl:free"

SCORE_THRESHOLD = 0.40
TOP_K = 5

REFUSAL_MARKERS = [
    "i don't have official information",
    "i do not have official information",
    "je n'ai pas d'information officielle",
    "no official information",
    "please consult the algerian consulate",
    "veuillez consulter le consulat",
]

_index = None
_meta = None


def load_index():
    global _index, _meta
    if _index is None:
        _index = TurboQuantIndex.load(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    return _index, _meta


def detect_language(text: str) -> str:
    """Crude language detection: French if common French words present."""
    french_markers = [
        "quel", "quelle", "quels", "quelles",
        "comment", "combien",
        "puis-je", "est-ce", "est ce",
        "déclaration", "monnaie", "seuil",
        "valeur", "devise", "douane",
        "voyageur", "autorisation",
        "est-il", "est-il permis",
    ]
    lower = text.lower()
    if any(m in lower for m in french_markers):
        return "fr"
    return "en"


def is_refusal(answer_text: str) -> bool:
    """Return True if the LLM's answer contains a refusal phrase."""
    lower = answer_text.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def retrieve(query: str, k: int = TOP_K):
    index, meta = load_index()
    qvec = embed([query]).astype(np.float32)
    scores, slots = index.search(qvec, k=k)

    results = []
    for score, slot in zip(scores[0], slots[0]):
        m = meta[int(slot)]
        results.append({
            "score": float(score),
            "text": m["text"],
            "doc_id": m["doc_id"],
            "authority": m["authority"],
            "language": m["language"],
            "source_url": m["source_url"],
            "source_name": m["source_name"],
            "heading_path": m["heading_path"],
            "is_translation": m["is_translation"],
            "translation_of": m["translation_of"],
            "conflicts": m["conflicts"],
        })
    return results


def deduplicate_by_language(results: list[dict], query_lang: str) -> list[dict]:
    """
    Fix 2: For each doc_id that has multiple language variants in the results,
    keep the variant matching the query language if available.
    Fall back to the highest-scoring variant otherwise.
    """
    by_doc: dict[str, list[dict]] = {}
    for r in results:
        by_doc.setdefault(r["doc_id"], []).append(r)

    kept: list[dict] = []
    for _doc_id, variants in by_doc.items():
        matching = [v for v in variants if v["language"] == query_lang]
        if matching:
            kept.extend(matching)
        else:
            kept.append(max(variants, key=lambda x: x["score"]))

    kept.sort(key=lambda x: x["score"], reverse=True)
    return kept


def build_prompt(query: str, chunks: list[dict], query_lang: str) -> str:
    """Fix 1: Explicitly instruct the LLM to answer in the query language."""
    lang_instruction = {
        "en": "You MUST answer in English, regardless of the language of the sources.",
        "fr": "Tu DOIS répondre en français, quelle que soit la langue des sources.",
    }.get(
        query_lang,
        "Answer in the same language as the question.",
    )

    sources_text = ""
    for i, c in enumerate(chunks, 1):
        sources_text += (
            f"\n[{i}] (authority={c['authority']}, source={c['source_name']}, "
            f"lang={c['language']})\n{c['text']}\n"
        )

    return f"""You are Marhba, a travel regulation assistant for Algeria.

{lang_instruction}

Answer the user's question using ONLY the sources below.

Rules:
1. If the sources do not contain the answer, say: "I don't have official information on this. Please consult the Algerian consulate." (or the equivalent in the answer language).
2. If sources disagree on a factual value (e.g., currency thresholds), prefer the source with authority "official-gov" and mention that other sources may be outdated.
3. Never present a foreign government advisory as Algerian law. If a source is "official-advisory", say "According to the UK government's advisory..." (or similar) rather than "Algerian law says...".
4. Cite the source number in brackets, e.g., [1].
5. Do not invent facts. Do not use knowledge outside the provided sources.

SOURCES:
{sources_text}

QUESTION: {query}

ANSWER:"""


async def call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(2):  # retry once on 5xx (transient upstream errors)
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/thewhitesword/marhba",
                    "X-Title": "Marhba",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if response.status_code >= 500 and attempt == 0:
                continue  # retry once on 5xx
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    # If both attempts returned 5xx, raise the last response's error
    response.raise_for_status()


async def answer(query: str) -> dict:
    results = retrieve(query)

    if not results or max(r["score"] for r in results) < SCORE_THRESHOLD:
        return {
            "answer": "I don't have official information on this. Please consult the Algerian consulate.",
            "sources": [],
            "refused": True,
        }

    query_lang = detect_language(query)
    results = deduplicate_by_language(results, query_lang)

    prompt = build_prompt(query, results, query_lang)
    answer_text = await call_openrouter(prompt)

    sources = [
        {
            "doc_id": r["doc_id"],
            "authority": r["authority"],
            "language": r["language"],
            "source_url": r["source_url"],
            "source_name": r["source_name"],
            "heading": r["heading_path"],
            "score": round(r["score"], 3),
        }
        for r in results
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "refused": is_refusal(answer_text),
    }