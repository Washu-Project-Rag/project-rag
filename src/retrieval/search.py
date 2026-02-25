import os
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/index/faiss.index"
META_PATH = "data/index/chunks_meta.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_meta(path: str) -> List[Dict]:
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    return v


def search(query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Missing index: {INDEX_PATH}. Run build_index.py first.")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing metadata: {META_PATH}. Run build_index.py first.")

    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    qv = embed_query(model, query)

    scores, idxs = index.search(qv, top_k)
    results = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0 or i >= len(meta):
            continue
        results.append((float(score), meta[i]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5, help="Top-k")
    args = ap.parse_args()

    results = search(args.q, top_k=args.k)

    print(f"\nQuery: {args.q}\nTop-{args.k} results:\n")
    for rank, (score, m) in enumerate(results, 1):
        title = m.get("title", "")
        sec = " > ".join(m.get("section_path", []) or [])
        cid = m.get("chunk_id", "")
        text = (m.get("text", "") or "").replace("\n", " ")
        if len(text) > 240:
            text = text[:240] + "..."

        print(f"[{rank}] score={score:.4f}  chunk_id={cid}")
        print(f"     title={title}")
        print(f"     section={sec}")
        print(f"     text={text}\n")


if __name__ == "__main__":
    main()