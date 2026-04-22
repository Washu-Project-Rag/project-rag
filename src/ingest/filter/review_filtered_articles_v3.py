import json
import random
from collections import Counter
from pathlib import Path

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles_v3.jsonl"
OUTPUT_SAMPLE_JSONL = PROJECT_ROOT / "data" / "processed" / "wiki_cs_v3_sample_for_review.jsonl"
OUTPUT_SAMPLE_TXT = PROJECT_ROOT / "data" / "processed" / "wiki_cs_v3_sample_for_review.txt"
OUTPUT_TOP_TERMS_TXT = PROJECT_ROOT / "data" / "processed" / "wiki_cs_v3_top_matched_terms.txt"

SAMPLE_SIZE = 100
RANDOM_SEED = 42
TEXT_PREVIEW_CHARS = 500


def reservoir_sample_jsonl(path, sample_size, seed=42):
    random.seed(seed)
    sample = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            article = json.loads(line)

            if len(sample) < sample_size:
                sample.append(article)
            else:
                j = random.randint(1, i)
                if j <= sample_size:
                    sample[j - 1] = article

    return sample


def save_sample_jsonl(sample, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for article in sample:
            row = {
                "id": article.get("id"),
                "title": article.get("title"),
                "url": article.get("url"),
                "cs_filter_score_v3": article.get("cs_filter_score_v3"),
                "title_core_v3": article.get("title_core_v3", []),
                "first_core_v3": article.get("first_core_v3", []),
                "rest_core_v3": article.get("rest_core_v3", []),
                "text_preview": article.get("text", "")[:TEXT_PREVIEW_CHARS],
                "manual_label": "",   # cs / related / non_cs
                "notes": ""
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sample_txt(sample, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, article in enumerate(sample, 1):
            f.write(f"[{i}]\n")
            f.write(f"ID: {article.get('id', '')}\n")
            f.write(f"Title: {article.get('title', '')}\n")
            f.write(f"Score v3: {article.get('cs_filter_score_v3', '')}\n")
            f.write(f"Title core v3: {article.get('title_core_v3', [])}\n")
            f.write(f"First core v3: {article.get('first_core_v3', [])}\n")
            f.write(f"Rest core v3: {article.get('rest_core_v3', [])}\n")
            f.write(f"URL: {article.get('url', '')}\n")
            f.write("Preview:\n")
            f.write(article.get("text", "")[:TEXT_PREVIEW_CHARS])
            f.write("\n\n")
            f.write("Manual label: ____________________\n")
            f.write("Notes: ___________________________\n")
            f.write("\n" + "=" * 80 + "\n\n")


def save_top_terms_streaming(input_path, output_path, top_k=200):
    counter = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            for term in article.get("title_core_v3", []):
                counter[f"title::{term}"] += 1
            for term in article.get("first_core_v3", []):
                counter[f"first::{term}"] += 1
            for term in article.get("rest_core_v3", []):
                counter[f"rest::{term}"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Top matched core terms in wiki_cs_articles_v3.jsonl\n")
        f.write("=" * 60 + "\n\n")
        for term, count in counter.most_common(top_k):
            f.write(f"{term}\t{count}\n")


def main():
    sample = reservoir_sample_jsonl(INPUT_FILE, SAMPLE_SIZE, seed=RANDOM_SEED)

    save_sample_jsonl(sample, OUTPUT_SAMPLE_JSONL)
    save_sample_txt(sample, OUTPUT_SAMPLE_TXT)
    save_top_terms_streaming(INPUT_FILE, OUTPUT_TOP_TERMS_TXT, top_k=200)

    print(f"Saved review JSONL to: {OUTPUT_SAMPLE_JSONL}")
    print(f"Saved review TXT to:   {OUTPUT_SAMPLE_TXT}")
    print(f"Saved top terms to:    {OUTPUT_TOP_TERMS_TXT}")


if __name__ == "__main__":
    main()