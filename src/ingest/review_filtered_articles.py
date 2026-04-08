import json
import random
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # src/ingest
PROJECT_ROOT = BASE_DIR.parent.parent               # project root

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles.jsonl"
OUTPUT_SAMPLE_JSONL = PROJECT_ROOT / "data" / "processed" / "wiki_cs_sample_for_review.jsonl"
OUTPUT_SAMPLE_TXT = PROJECT_ROOT / "data" / "processed" / "wiki_cs_sample_for_review.txt"
OUTPUT_TOP_TERMS_TXT = PROJECT_ROOT / "data" / "processed" / "wiki_cs_top_matched_terms.txt"

SAMPLE_SIZE = 100
RANDOM_SEED = 42
TEXT_PREVIEW_CHARS = 500


def load_articles(path):
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def save_sample_jsonl(sample, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for article in sample:
            row = {
                "id": article.get("id"),
                "title": article.get("title"),
                "url": article.get("url"),
                "cs_filter_score": article.get("cs_filter_score"),
                "matched_terms": article.get("matched_terms", []),
                "text_preview": article.get("text", "")[:TEXT_PREVIEW_CHARS],
                "manual_label": "",   # cs / borderline / non_cs
                "notes": ""
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sample_txt(sample, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, article in enumerate(sample, 1):
            f.write(f"[{i}]\n")
            f.write(f"ID: {article.get('id', '')}\n")
            f.write(f"Title: {article.get('title', '')}\n")
            f.write(f"Score: {article.get('cs_filter_score', '')}\n")
            f.write(f"Matched terms: {article.get('matched_terms', [])}\n")
            f.write(f"URL: {article.get('url', '')}\n")
            f.write("Preview:\n")
            f.write(article.get("text", "")[:TEXT_PREVIEW_CHARS])
            f.write("\n\n")
            f.write("Manual label: ____________________\n")
            f.write("Notes: ___________________________\n")
            f.write("\n" + "=" * 80 + "\n\n")


def save_top_terms(articles, output_path, top_k=200):
    counter = Counter()

    for article in articles:
        for term in article.get("matched_terms", []):
            counter[term] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Top matched terms in wiki_cs_articles.jsonl\n")
        f.write("=" * 50 + "\n\n")
        for term, count in counter.most_common(top_k):
            f.write(f"{term}\t{count}\n")


def main():
    random.seed(RANDOM_SEED)

    articles = load_articles(INPUT_FILE)
    print(f"Loaded articles: {len(articles)}")

    sample_size = min(SAMPLE_SIZE, len(articles))
    sample = random.sample(articles, sample_size)

    save_sample_jsonl(sample, OUTPUT_SAMPLE_JSONL)
    save_sample_txt(sample, OUTPUT_SAMPLE_TXT)
    save_top_terms(articles, OUTPUT_TOP_TERMS_TXT, top_k=200)

    print(f"Saved review JSONL to: {OUTPUT_SAMPLE_JSONL}")
    print(f"Saved review TXT to:   {OUTPUT_SAMPLE_TXT}")
    print(f"Saved top terms to:    {OUTPUT_TOP_TERMS_TXT}")


if __name__ == "__main__":
    main()