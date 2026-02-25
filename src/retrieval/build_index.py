import os
import json
from typing import List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


INPUT_PATH = "data/processed/corpus_chunks.jsonl"
OUT_DIR = "data/index"
INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")
META_PATH = os.path.join(OUT_DIR, "chunks_meta.jsonl")

# A strong default embedding model for English retrieval
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: str) -> List[Dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def make_embed_text(c: Dict) -> str:
    title = c.get("title", "")
    sec = " > ".join(c.get("section_path", []) or [])
    text = c.get("text", "")
    return f"Title: {title}\nSection: {sec}\nContent: {text}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    chunks = load_chunks(INPUT_PATH)
    if not chunks:
        raise ValueError(f"No chunks found in {INPUT_PATH}")

    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [make_embed_text(c) for c in chunks]

    # Embed in batches
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    # Save metadata aligned with FAISS vector order
    with open(META_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            meta = {
                "chunk_id": c["chunk_id"],
                "title": c.get("title", ""),
                "section_path": c.get("section_path", []),
                "paragraph_index": c.get("paragraph_index", None),
                "text": c.get("text", ""),
                "source": c.get("source", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Saved FAISS index: {INDEX_PATH}")
    print(f"Saved metadata:   {META_PATH}")
    print(f"Vectors: {index.ntotal}, dim: {dim}")


if __name__ == "__main__":
    main()