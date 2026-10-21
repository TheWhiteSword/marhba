# Marhba Corpus — Source Index

This is the human-readable index of the 8 official sources used by Marhba. Each row corresponds to a `.md` file in this directory (or its `.fr` / `.en` variant for bilingual sources).

## Corpus Table

| # | `doc_id` | Title | Authority | Applicability | Lang | Updated | Source URL |
|---|---|---|---|---|---|---|---|
| 1 | `douane-2024-traveler-leaflet` | For your trip to Algeria — Useful information for travelers | `official-gov` | universal | en | 2024-06-12 | [douane.gov.dz PDF](https://douane.gov.dz/IMG/pdf/depliant_se_2024_infos_ang_07_.pdf) |
| 2 | `douane-currency-declaration-2026` | La Déclaration de la monnaie | `official-gov` | universal | fr + en | UNKNOWN | [douane.gov.dz](https://douane.gov.dz/spip.php?article310) |
| 3 | `mfa-entry-visa-2026` | Entry visa to Algeria | `official-gov` | universal | en | UNKNOWN | [mfa.gov.dz](https://www.mfa.gov.dz/services-for-foreigners/entry-visa-to-algeria) |
| 4 | `bank-algeria-reg-07-01-2007` | Règlement n°07-01 du 09/01/2007 | `official-gov` | universal | fr + en | 2007-01-09 | [bank-of-algeria.dz](https://www.bank-of-algeria.dz/) |
| 5 | `dz-consulate-london-visa-2026` | Consulate General of Algeria in London — Entry visa | `official-gov` | universal | en | UNKNOWN | [cglondon.mfa.gov.dz](https://cglondon.mfa.gov.dz/en/services-for-foreigners/entry-visa-to-algeria) |
| 6 | `dz-embassy-beijing-visa-2026` | Embassy of Algeria in Beijing — Entry visa | `official-gov` | nationality:CN | en | UNKNOWN | [embbeijing.mfa.gov.dz](https://embbeijing.mfa.gov.dz/en/services-for-foreigners/entry-visa-to-algeria-1) |
| 7 | `uk-fcdo-entry-req-2026` | Entry requirements — Algeria travel advice | `official-advisory` | nationality:UK | en | 2026-09-25 | [gov.uk](https://www.gov.uk/foreign-travel-advice/algeria/entry-requirements) |
| 8 | `uk-fcdo-safety-2026` | Safety and security — Algeria travel advice | `official-advisory` | nationality:UK | en | 2026-09-25 | [gov.uk](https://www.gov.uk/foreign-travel-advice/algeria/safety-and-security) |

## Authority Levels

| Level | Meaning | Used for |
|---|---|---|
| `official-gov` | Algerian government source | Highest trust. Takes precedence on conflicts. |
| `official-advisory` | Foreign government advisory | Authoritative interpretation, not Algerian law. |
| `secondary` | News, blogs, forums | Not used in v1. Planned for v2. |

## Language Handling

- Sources 2 and 4 exist in both French and English (`.fr.md` and `.en.md` files)
- All others are English-only
- Cross-lingual retrieval is handled by Gemini's multilingual embeddings — a French query retrieves French chunks, an English query retrieves English chunks

## Known Conflicts

| Topic | Source A | Value | Source B | Value | Resolution |
|---|---|---|---|---|---|
| Currency declaration threshold | `douane-currency-declaration-2026` (official-gov) | 1,000 EUR | `uk-fcdo-entry-req-2026` (official-advisory) | 5,000 EUR | Prefer official-gov; flag advisory as possibly outdated |

## Known Gaps

Topics the corpus does **not** cover, and the system will refuse to answer:

- **Personal medication rules** — no official Algerian source found
- **Visa exemptions by nationality** — only stale (2017) airport data found; excluded

## File Naming Convention

Each source is one markdown file with YAML front-matter:

```
data/sources/
├── douane-2024-traveler-leaflet.md
├── douane-currency-declaration-2026.fr.md
├── douane-currency-declaration-2026.en.md
├── mfa-entry-visa-2026.md
├── bank-algeria-reg-07-01-2007.fr.md
├── bank-algeria-reg-07-01-2007.en.md
├── dz-consulate-london-visa-2026.md
├── dz-embassy-beijing-visa-2026.md
├── uk-fcdo-entry-req-2026.md
└── uk-fcdo-safety-2026.md
```

Bilingual sources use `.fr.md` (original) and `.en.md` (translation, with `translation_of` metadata).