import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # src/ingest
PROJECT_ROOT = BASE_DIR.parent.parent               # project root

KEPT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles_token_v2.jsonl"
INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "wiki_extracted"
OUTPUT_SAMPLE_JSONL = PROJECT_ROOT / "data" / "processed" / "wiki_rejected_sample_for_review_with_token.jsonl"
# OUTPUT_SAMPLE_TXT = PROJECT_ROOT / "data" / "processed" / "wiki_rejected_sample_for_review_with_token.txt"

SAMPLE_SIZE = 100
RANDOM_SEED = 42
TEXT_PREVIEW_CHARS = 500


def load_kept_ids(path: Path) -> set:
    kept_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            article_id = article.get("id")
            if article_id is not None:
                kept_ids.add(article_id)
    return kept_ids


def iter_wiki_files(input_root: Path):
    for path in sorted(input_root.rglob("wiki_*")):
        if path.is_file():
            yield path


def collect_rejected_articles(input_root: Path, kept_ids: set) -> list[dict]:
    rejected = []

    for file_path in iter_wiki_files(input_root):
        print(f"Processing: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                article = json.loads(line)
                article_id = article.get("id")

                # 如果 id 不在 kept_ids 中，就视为 rejected
                if article_id not in kept_ids:
                    row = {
                        "id": article.get("id"),
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "text_preview": article.get("text", "")[:TEXT_PREVIEW_CHARS],
                        "manual_label": "",   # cs / related / non_cs
                        "notes": ""
                    }
                    rejected.append(row)

    return rejected


def save_sample_jsonl(sample: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sample_txt(sample: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, article in enumerate(sample, 1):
            f.write(f"[{i}]\n")
            f.write(f"ID: {article.get('id', '')}\n")
            f.write(f"Title: {article.get('title', '')}\n")
            f.write(f"URL: {article.get('url', '')}\n")
            f.write("Preview:\n")
            f.write(article.get("text_preview", ""))
            f.write("\n\n")
            f.write("Manual label: ____________________\n")
            f.write("Notes: ___________________________\n")
            f.write("\n" + "=" * 80 + "\n\n")


def main():
    random.seed(RANDOM_SEED)

    print("Loading kept article IDs...")
    kept_ids = load_kept_ids(KEPT_FILE)
    print(f"Loaded kept IDs: {len(kept_ids)}")

    print("Collecting rejected articles by set difference...")
    rejected_articles = collect_rejected_articles(INPUT_ROOT, kept_ids)
    print(f"Total rejected articles found: {len(rejected_articles)}")

    sample_size = min(SAMPLE_SIZE, len(rejected_articles))
    sample = random.sample(rejected_articles, sample_size)

    OUTPUT_SAMPLE_JSONL.parent.mkdir(parents=True, exist_ok=True)

    save_sample_jsonl(sample, OUTPUT_SAMPLE_JSONL)
    #save_sample_txt(sample, OUTPUT_SAMPLE_TXT)

    print(f"Saved rejected review JSONL to: {OUTPUT_SAMPLE_JSONL}")
    #print(f"Saved rejected review TXT to:   {OUTPUT_SAMPLE_TXT}")


if __name__ == "__main__":
    main()