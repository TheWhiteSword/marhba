"""
Marhba — Build vector index from markdown corpus.

Usage:
    python build_index.py

Input:
    data/sources/*.md   (with YAML front-matter)

Output:
    data/index.tv       (Turbovec TurboQuantIndex)
    data/index_meta.json (chunk metadata, aligned by slot position)
"""

import json
import re
from pathlib import Path
from typing import Any

import frontmatter
import numpy as np
from turbovec import TurboQuantIndex

from app.embed import embed

# ---- Config ----
SOURCES_DIR = Path("data/sources")
INDEX_PATH = Path("data/index.tv")
META_PATH = Path("data/index_meta.json")
EMBED_DIM = 768  # Gemini gemini-embedding-001 output dimension
BIT_WIDTH = 4  # 2, 3, or 4. 4 = best accuracy.
MAX_CHUNK_CHARS = 500
MIN_CHUNK_CHARS = 100


def parse_markdown_file(path: Path) -> tuple[dict[str, Any], str]:
    """Parse YAML front-matter and return (metadata, body)."""
    post = frontmatter.load(path)
    return dict(post.metadata), post.content


def split_into_sections(body: str) -> list[dict[str, str]]:
    """
    Split markdown body into sections by headings.
    Returns list of {"heading_path": str, "content": str}.
    """
    lines = body.split("\n")
    sections: list[dict[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            path_str = " > ".join(h[1] for h in heading_stack) if heading_stack else ""
            sections.append({"heading_path": path_str, "content": content})
        buffer.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level = len(m.group(1))
            text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
        else:
            buffer.append(line)

    flush()
    return sections


def chunk_section(section: dict[str, str]) -> list[str]:
    """
    Split a section into chunks of <= MAX_CHUNK_CHARS.
    Prepends heading path to each chunk for retrieval context.
    """
    path = section["heading_path"]
    content = section["content"].strip()

    if len(content) <= MAX_CHUNK_CHARS:
        text = f"{path} — {content}" if path else content
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    for p in paragraphs:
        if len(buf) + len(p) + 2 <= MAX_CHUNK_CHARS:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                text = f"{path} — {buf}" if path else buf
                if len(text) >= MIN_CHUNK_CHARS:
                    chunks.append(text)
            if len(p) > MAX_CHUNK_CHARS:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                sub_buf = ""
                for s in sentences:
                    if len(sub_buf) + len(s) + 1 <= MAX_CHUNK_CHARS:
                        sub_buf = f"{sub_buf} {s}" if sub_buf else s
                    else:
                        if sub_buf:
                            text = f"{path} — {sub_buf}" if path else sub_buf
                            if len(text) >= MIN_CHUNK_CHARS:
                                chunks.append(text)
                        sub_buf = s
                buf = sub_buf
            else:
                buf = p

    if buf:
        text = f"{path} — {buf}" if path else buf
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(text)

    return chunks


def build_chunk_records() -> list[dict[str, Any]]:
    """Walk data/sources/*.md, parse, chunk, return records with metadata."""
    records: list[dict[str, Any]] = []
    md_files = sorted(SOURCES_DIR.glob("*.md"))

    if not md_files:
        raise SystemExit(f"No .md files found in {SOURCES_DIR}")

    print(f"Found {len(md_files)} markdown files")

    for path in md_files:
        meta, body = parse_markdown_file(path)
        sections = split_into_sections(body)

        doc_id = meta.get("doc_id", path.stem)
        is_translation = bool(meta.get("translation_of"))

        chunk_count = 0
        for section in sections:
            chunks = chunk_section(section)
            for chunk_text in chunks:
                records.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "file": path.name,
                    "heading_path": section["heading_path"],
                    "authority": meta.get("authority", "UNKNOWN"),
                    "applicability": meta.get("applicability", "UNKNOWN"),
                    "jurisdiction": meta.get("jurisdiction", "UNKNOWN"),
                    "language": meta.get("language", "UNKNOWN"),
                    "source_url": meta.get("source_url", ""),
                    "source_name": meta.get("source_name", ""),
                    "is_translation": is_translation,
                    "translation_of": meta.get("translation_of"),
                    "conflicts": meta.get("conflicts", []),
                })
                chunk_count += 1

        print(f"  {path.name}: {len(sections)} sections, {chunk_count} chunks")

    return records


def main():
    print("=== Marhba index builder ===\n")

    print("[1/4] Parsing and chunking sources...")
    records = build_chunk_records()
    print(f"Total chunks: {len(records)}\n")

    if not records:
        raise SystemExit("No chunks produced. Check your source files.")

    print(f"[2/4] Embedding {len(records)} chunks via Gemini API...")
    print("      This will make one API call per chunk. Be patient.")
    texts = [r["text"] for r in records]

    # Embed in batches of 20 to avoid overwhelming the API and to give progress
    all_vectors = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = embed(batch)
        all_vectors.append(vecs)
        print(f"      Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    vectors = np.vstack(all_vectors).astype(np.float32)
    print(f"Embeddings shape: {vectors.shape}\n")

    if vectors.shape[1] != EMBED_DIM:
        raise SystemExit(
            f"Expected {EMBED_DIM}-dim embeddings, got {vectors.shape[1]}. "
            f"Update EMBED_DIM in build_index.py."
        )

    print(f"[3/4] Building Turbovec index (dim={EMBED_DIM}, bit_width={BIT_WIDTH})")
    index = TurboQuantIndex(dim=EMBED_DIM, bit_width=BIT_WIDTH)
    index.add(np.ascontiguousarray(vectors))
    index.write(str(INDEX_PATH))
    print(f"Index written to {INDEX_PATH}")

    print("[4/4] Saving metadata...")
    meta_records = []
    for i, r in enumerate(records):
        meta_records.append({
            "slot": i,
            "text": r["text"],
            "doc_id": r["doc_id"],
            "file": r["file"],
            "heading_path": r["heading_path"],
            "authority": r["authority"],
            "applicability": r["applicability"],
            "jurisdiction": r["jurisdiction"],
            "language": r["language"],
            "source_url": r["source_url"],
            "source_name": r["source_name"],
            "is_translation": r["is_translation"],
            "translation_of": r["translation_of"],
            "conflicts": r["conflicts"],
        })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, indent=2, ensure_ascii=False)
    print(f"Metadata written to {META_PATH}")

    print("\n=== Done ===")
    print(f"Chunks indexed: {len(records)}")
    print(f"Index file size: {INDEX_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Metadata file size: {META_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()