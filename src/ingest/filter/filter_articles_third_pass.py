import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles_v2.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles_v3.jsonl"

CORE_TERMS = [
    "computer science",
    "programming language",
    "operating system",
    "database",
    "compiler",
    "cryptography",
    "computer security",
    "machine learning",
    "artificial intelligence",
    "computer network",
    "software engineering",
    "data structure",
    "algorithm",
    "computer programming",
    "computer program",
    "source code",
    "formal methods",
    "distributed computing",
    "information retrieval",
    "computer vision",
    "computer graphics",
]

# 相关但更宽：只能辅助
SUPPORT_TERMS = [
    "software",
    "interface",
    "computing",
    "open-source software",
    "user interface",
    "technical documentation",
    "operating system",
    "kernel",
    "virtual machine",
    "programming language",
    "database",
    "computer science",
]

# 明显容易误伤：不单独作为依据
BLOCK_OR_WEAK_TERMS = {
    "java",
    "node",
    "prolog",
    "variable",
    "ascii",
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_paragraphs(text: str):
    return [p.strip() for p in text.split("\n") if p.strip()]

def contains_term_whole_word(text: str, term: str) -> bool:
    pattern = r"\b" + re.escape(term.lower()) + r"\b"
    return re.search(pattern, text) is not None

def matched_terms(text: str, terms: list[str]) -> list[str]:
    hits = []
    for term in terms:
        if contains_term_whole_word(text, term):
            hits.append(term)
    return hits

def score_article(title: str, text: str):
    title_l = normalize_text(title)
    paragraphs = split_paragraphs(text)
    first_part = normalize_text(" ".join(paragraphs[:2]))
    rest_part = normalize_text(" ".join(paragraphs[2:]))

    title_core = matched_terms(title_l, CORE_TERMS)
    first_core = matched_terms(first_part, CORE_TERMS)
    rest_core = matched_terms(rest_part, CORE_TERMS)

    title_support = matched_terms(title_l, SUPPORT_TERMS)
    first_support = matched_terms(first_part, SUPPORT_TERMS)
    rest_support = matched_terms(rest_part, SUPPORT_TERMS)

    # 去掉弱误伤词
    title_core = [t for t in title_core if t not in BLOCK_OR_WEAK_TERMS]
    first_core = [t for t in first_core if t not in BLOCK_OR_WEAK_TERMS]
    rest_core = [t for t in rest_core if t not in BLOCK_OR_WEAK_TERMS]

    score = 0
    score += 5 * len(set(title_core))
    score += 3 * len(set(first_core))
    score += 1 * len(set(rest_core))
    score += 1 * len(set(title_support))
    score += 1 * len(set(first_support))

    details = {
        "title_core": sorted(set(title_core)),
        "first_core": sorted(set(first_core)),
        "rest_core": sorted(set(rest_core)),
        "title_support": sorted(set(title_support)),
        "first_support": sorted(set(first_support)),
    }
    return score, details

def keep_article(score: int, details: dict) -> bool:
    # 规则尽量简单
    if len(details["title_core"]) >= 1:
        return True
    if len(details["first_core"]) >= 2:
        return True
    if (len(details["title_core"]) + len(details["first_core"]) + len(details["rest_core"])) >= 2 and score >= 6:
        return True
    return False

def main():
    total = 0
    kept = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            article = json.loads(line)

            title = article.get("title", "")
            text = article.get("text", "")
            if not text.strip():
                continue

            score, details = score_article(title, text)

            if keep_article(score, details):
                article["cs_filter_score_v3"] = score
                article["title_core_v3"] = details["title_core"]
                article["first_core_v3"] = details["first_core"]
                article["rest_core_v3"] = details["rest_core"]
                fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                kept += 1

            if total % 100000 == 0:
                print(f"Processed {total:,} | kept {kept:,}")

    print(f"Total seen: {total:,}")
    print(f"Kept v3: {kept:,}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()