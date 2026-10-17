# Marhba

A retrieval-augmented generation (RAG) system for Algerian travel regulations. Answers tourist questions grounded in official sources.

## Problem

Tourists arrive in Algeria and unknowingly break rules they didn't know existed. Customs limits, visa requirements, currency rules, photography restrictions — this information is scattered across government PDFs, embassy pages, and outdated forum posts. Marhba gives straight, grounded answers from official sources.

## Status

Early prototype. Corpus collection complete; pipeline in development.

## Architecture

```
Question → Embed (bekko-a8m) → Retrieve (Turbovec) → Generate (LLM API) → Answer + citations
```

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

See `source_tracker.csv` for full metadata.

## Authority Model

Sources are tagged with two dimensions:

- **`authority`**: `official-gov` (Algerian government) > `official-advisory` (foreign government) > `secondary`
- **`applicability`**: `universal` (all travelers) or `nationality:XX` (country-specific)

On conflicts, `official-gov` sources take precedence.

## Stack

- **Embedding**: `kyr0/bekko-embedding-v1-a8m` (multilingual, 8M params, CPU-friendly)
- **Vector store**: Turbovec
- **LLM**: API-based (Gemini Flash / Groq / DeepSeek)
- **Hosting**: Render free tier (native Python)

## Design Principles

- **Grounded answers only.** If the information isn't in the corpus, the system says "I don't know" rather than guessing.
- **Honest attribution.** Answers cite their source and authority level.
- **Bilingual.** French sources are preserved alongside English translations. Cross-lingual retrieval works via the multilingual embedding model.
- **Conflict-aware.** When sources disagree, the system surfaces both and prefers `official-gov`.

## Local Development

```bash
pip install -r requirements.txt
python build_index.py        # builds data/index.tvim
uvicorn app.main:app --reload  # runs the API locally
```

## License

MIT