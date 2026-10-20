# Marhba

A RAG assistant that gives tourists grounded answers about Algerian travel regulations, citing official sources and refusing when it doesn't know.

**Live demo:** https://marhba.onrender.com

> **Note:** Deployed on Render's free tier. The first request after 15+ minutes of inactivity takes 30–60 seconds to wake up. The frontend shows a "Waking up the server" banner during this window. Subsequent requests are fast (2–5 seconds).

## Problem

Tourists arrive in Algeria and unknowingly break rules they didn't know existed. Customs limits, visa requirements, currency declarations, photography restrictions — this information is scattered across government PDFs, embassy pages, and outdated forum posts, mostly in French or Arabic. Generic chatbots answer confidently but invent specifics, which for a regulatory tool can mean fines or detention.

Marhba is a retrieval-augmented generation (RAG) system that answers travel-regulation questions grounded strictly in 8 official sources. Every answer cites its source, flags conflicts between sources, and refuses when the corpus doesn't cover the question.

## Architecture

```
Question → Gemini embed (768-dim) → Turbovec retrieve → OpenRouter LLM → Answer + citations
```

1. **Embed the query** with Google Gemini's `gemini-embedding-001` (768 dimensions)
2. **Retrieve top-5 chunks** from a Turbovec index (169 chunks from 8 sources, 4-bit quantized)
3. **Deduplicate by language** — if both French and English versions of the same source are retrieved, keep the one matching the query language
4. **Build a grounded prompt** with source authority metadata and conflict-resolution rules
5. **Generate the answer** via OpenRouter, enforcing citations and honest refusals

## Corpus

| # | Source | Authority | Language |
|---|---|---|---|
| 1 | Algerian Customs 2024 traveler leaflet | official-gov | en |
| 2 | Algerian Customs currency declaration | official-gov | fr + en |
| 3 | Algerian MFA entry visa | official-gov | en |
| 4 | Bank of Algeria Regulation 07-01 | official-gov | fr + en |
| 5 | Algerian Consulate London visa | official-gov | en |
| 6 | Algerian Embassy Beijing visa | official-gov | en |
| 7 | UK FCDO entry requirements | official-advisory | en |
| 8 | UK FCDO safety and security | official-advisory | en |

See [`data/sources/SOURCES.md`](data/sources/SOURCES.md) for full metadata, language handling, known conflicts, and gaps.

## Authority Model

Sources are tagged with two dimensions:

- **`authority`**: `official-gov` (Algerian government) > `official-advisory` (foreign government) > `secondary`
- **`applicability`**: `universal` (all travelers) or `nationality:XX` (country-specific)

On conflicts, `official-gov` sources take precedence. For example, the currency declaration threshold is **1,000 EUR** per Algerian Customs (source #2), while the UK advisory (source #7) says 5,000 EUR. The system prefers the Algerian source and flags the discrepancy.

## Stack

- **Embedding**: Google Gemini `gemini-embedding-001` (768-dim, cloud API)
- **Vector store**: Turbovec (`TurboQuantIndex`, 4-bit quantization)
- **LLM**: OpenRouter — `nvidia/nemotron-3.5-super-vl:free`
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla HTML/JS with Algerian flag-inspired CSS, served via FastAPI `StaticFiles`
- **Hosting**: Render free tier (native Python)

## Design Principles

- **Grounded answers only.** If the information isn't in the corpus, the system says "I don't have official information on this" rather than guessing.
- **Honest attribution.** Answers cite their source and authority level.
- **Bilingual.** French and English source versions are preserved. Cross-lingual retrieval works via Gemini's multilingual embeddings.
- **Conflict-aware.** When sources disagree, the system surfaces both and prefers `official-gov`.
- **Refusal detection.** Two layers: a score threshold (0.40) short-circuits low-confidence retrievals before calling the LLM; a second check on the LLM's answer text sets the `refused` flag when the model declines.

## Known Gaps

Deliberate refusals, not bugs:

- **Personal medication rules** — no official Algerian traveler-medication source in the corpus
- **Visa exemptions by nationality** — only stale (2017) data found; excluded rather than risking wrong answers

The system answers these questions with "I don't have official information on this. Please consult the Algerian consulate."

## Local Development

**Prerequisites:** Python 3.13, an OpenRouter API key, a Google Gemini API key.

```bash
# 1. Clone and enter the repo
git clone https://github.com/TheWhiteSword/marhba
cd marhba

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add:
#   OPENROUTER_API_KEY=sk-or-...
#   GEMINI_API_KEY=...

# 5. Build the vector index (one-time, ~3 minutes with Gemini rate limits)
python build_index.py

# 6. Run the app
uvicorn app.main:app --reload

# 7. Open http://127.0.0.1:8000/ in your browser
```

## Deployment Notes

This project was developed to run on Render's **free tier (512 MB RAM)**. That constraint drove several architectural decisions:

- **Local embeddings failed.** PyTorch + `sentence-transformers` consumed ~400 MB at load, causing OOM. Even ONNX Runtime with a 125 MB model exceeded the limit during inference.
- **Git LFS is not supported** in Render's native Python build environment (read-only filesystem prevents `apt-get install git-lfs`).
- **Cloud embeddings** via Gemini's free tier solve the memory problem entirely: the container uses ~150 MB, no local model, no cold-start delay from model loading.

The final architecture uses Gemini for embeddings and OpenRouter for generation — both free-tier compatible, both cloud APIs.

## Testing

A 17-question evaluation set lives in `tests/test_questions.md`, covering:

- Factual customs queries (tobacco, alcohol, pets)
- Conflict resolution (currency thresholds)
- Multilingual queries (English + French)
- Prohibited items (drones, sensitive equipment)
- Refusal cases (medication, visa exemptions by nationality)

## Team and Roles

**Solo project.** All work by Sid Ahmed Mahieddine:

- Corpus collection and curation (8 official sources, bilingual EN/FR)
- RAG pipeline architecture (chunking, embedding, retrieval, generation)
- Backend implementation (FastAPI, Gemini embeddings, OpenRouter LLM)
- Frontend implementation (Algerian flag-inspired chat UI)
- Deployment (Render free tier, cold-start mitigation)
- Testing and evaluation (17-question gold set)

## AI Tools Used

- **Claude** — architecture design, code generation, corpus curation
- **Gemini API** — embeddings at runtime
- **OpenRouter** — LLM inference at runtime

## License

MIT