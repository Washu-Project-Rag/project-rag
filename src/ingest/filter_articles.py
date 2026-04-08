import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # src/ingest
PROJECT_ROOT = BASE_DIR.parent.parent               # project root

INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "wiki_extracted"
GLOSSARY_FILE = BASE_DIR / "glossary_terms.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles.jsonl"


def load_glossary_terms(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t.lower().strip() for t in data["terms"] if t.strip()]


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_paragraphs(text):
    return [p.strip() for p in text.split("\n") if p.strip()]


def score_article(title, text, glossary_terms):
    title_l = normalize_text(title)
    paragraphs = split_paragraphs(text)

    first_part = normalize_text(" ".join(paragraphs[:2]))
    rest_part = normalize_text(" ".join(paragraphs[2:]))

    score = 0
    matched_terms = set()

    for term in glossary_terms:
        if term in title_l:
            score += 3
            matched_terms.add(term)

        if term in first_part:
            score += 2
            matched_terms.add(term)
        elif term in rest_part:
            score += 1
            matched_terms.add(term)

    return score, matched_terms


def keep_article(score, matched_terms):
    return (score >= 4) or (len(matched_terms) >= 2)


def iter_wiki_files(input_root):
    for path in sorted(input_root.rglob("wiki_*")):
        if path.is_file():
            yield path


def main():
    glossary_terms = load_glossary_terms(GLOSSARY_FILE)

    total_files = 0
    total_articles = 0
    kept_articles = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for file_path in iter_wiki_files(INPUT_ROOT):
            total_files += 1
            print(f"Processing: {file_path}")

            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    total_articles += 1
                    article = json.loads(line)

                    title = article.get("title", "")
                    text = article.get("text", "")

                    if not text.strip():
                        continue

                    score, matched_terms = score_article(title, text, glossary_terms)

                    if keep_article(score, matched_terms):
                        article["cs_filter_score"] = score
                        article["matched_terms"] = sorted(list(matched_terms))[:30]
                        article["source_file"] = str(file_path.relative_to(PROJECT_ROOT))
                        fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                        kept_articles += 1

    print(f"Total files processed: {total_files}")
    print(f"Total articles seen: {total_articles}")
    print(f"Kept CS articles: {kept_articles}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()