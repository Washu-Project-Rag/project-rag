import json
import re
from pathlib import Path

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parent          # src/ingest
PROJECT_ROOT = BASE_DIR.parent.parent               # project root

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wiki_cs_articles_v2.jsonl"


# =============================
# High-confidence CS terms only
# =============================
HIGH_CONFIDENCE_TERMS = sorted(set([
    "algorithm",
    "algorithm design",
    "application programming interface",
    "application software",
    "artificial intelligence",
    "ascii",
    "automata theory",
    "bandwidth",
    "benchmark",
    "binary number",
    "bioinformatics",
    "bit rate",
    "booting",
    "boolean algebra",
    "callback",
    "central processing unit",
    "cipher",
    "cloud computing",
    "coding theory",
    "compiler",
    "computation",
    "computability theory",
    "computational biology",
    "computational chemistry",
    "computational complexity theory",
    "computational model",
    "computational neuroscience",
    "computational physics",
    "computational science",
    "computer architecture",
    "computer graphics",
    "computer network",
    "computer programming",
    "computer program",
    "computer science",
    "computer scientist",
    "computer security",
    "computer vision",
    "computing",
    "concurrency",
    "control flow",
    "creative commons",
    "cryptography",
    "csv",
    "cyberspace",
    "data center",
    "data mining",
    "data science",
    "data structure",
    "data type",
    "database",
    "daemon",
    "debugging",
    "digital data",
    "digital signal processing",
    "distributed computing",
    "dns",
    "domain name system",
    "emulator",
    "encryption",
    "executable",
    "exception handling",
    "feasibility study",
    "filename extension",
    "floating-point arithmetic",
    "for loop",
    "formal methods",
    "functional programming",
    "game theory",
    "gigabyte",
    "graph theory",
    "hash function",
    "hash table",
    "human-computer interaction",
    "image processing",
    "information retrieval",
    "input/output",
    "integrated development environment",
    "intelligent agent",
    "interface",
    "interpreter",
    "iteration",
    "java",
    "kernel",
    "linker",
    "linked list",
    "loader",
    "logic programming",
    "machine learning",
    "machine vision",
    "mathematical logic",
    "matrix",
    "modem",
    "natural language processing",
    "node",
    "number theory",
    "numerical analysis",
    "numerical method",
    "object code",
    "object-oriented programming",
    "open-source software",
    "operating system",
    "optical fiber",
    "parallel computing",
    "parameter",
    "peripheral",
    "pointer",
    "programming language",
    "prolog",
    "python",
    "quantum computing",
    "queue",
    "r programming language",
    "radix",
    "recursion",
    "relational database",
    "robotics",
    "router",
    "run time",
    "search algorithm",
    "semantics",
    "serialization",
    "software",
    "software design",
    "software development",
    "software engineering",
    "software testing",
    "source code",
    "stack",
    "subroutine",
    "syntax",
    "technical documentation",
    "type theory",
    "user agent",
    "user interface",
    "user interface design",
    "variable",
    "virtual machine",
    "wi-fi",
    "xhtml",
]))


# =============================
# Optional weaker-but-still-useful terms
# Only count in title, not body
# =============================
TITLE_ONLY_TERMS = sorted(set([
    "computer",
    "computing",
    "database",
    "software",
    "compiler",
    "kernel",
    "algorithm",
    "programming language",
    "machine learning",
    "artificial intelligence",
    "computer graphics",
    "computer network",
    "computer vision",
    "data structure",
    "information retrieval",
    "operating system",
    "cryptography",
    "computer architecture",
]))


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]


def count_matches(text: str, terms: list[str]) -> set[str]:
    matched = set()
    for term in terms:
        if term in text:
            matched.add(term)
    return matched


def score_article(title: str, text: str):
    title_l = normalize_text(title)
    paragraphs = split_paragraphs(text)

    first_part = normalize_text(" ".join(paragraphs[:2]))
    rest_part = normalize_text(" ".join(paragraphs[2:]))

    score = 0
    matched_terms = set()

    # Strong signals
    title_matches = count_matches(title_l, HIGH_CONFIDENCE_TERMS)
    first_matches = count_matches(first_part, HIGH_CONFIDENCE_TERMS)
    rest_matches = count_matches(rest_part, HIGH_CONFIDENCE_TERMS)

    # Weaker title-only signals
    title_only_matches = count_matches(title_l, TITLE_ONLY_TERMS)

    # Scoring
    score += 5 * len(title_matches)
    score += 3 * len(first_matches)
    score += 1 * len(rest_matches)
    score += 1 * len(title_only_matches - title_matches)

    matched_terms.update(title_matches)
    matched_terms.update(first_matches)
    matched_terms.update(rest_matches)
    matched_terms.update(title_only_matches)

    return score, matched_terms, {
        "title_matches": sorted(title_matches),
        "first_matches": sorted(first_matches),
        "rest_matches": sorted(rest_matches),
        "title_only_matches": sorted(title_only_matches),
    }


def keep_article(score: int, details: dict) -> bool:
    title_hits = len(details["title_matches"])
    first_hits = len(details["first_matches"])
    total_strong_hits = title_hits + first_hits + len(details["rest_matches"])

    # Strict rules:
    # 1) strong title hit
    if title_hits >= 1:
        return True

    # 2) at least two strong hits in first paragraphs
    if first_hits >= 2:
        return True

    # 3) enough overall strong evidence
    if total_strong_hits >= 3 and score >= 6:
        return True

    return False


def main():
    total_articles = 0
    kept_articles = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            total_articles += 1
            article = json.loads(line)

            title = article.get("title", "")
            text = article.get("text", "")

            if not text.strip():
                continue

            score, matched_terms, details = score_article(title, text)

            if keep_article(score, details):
                article["cs_filter_score_v2"] = score
                article["matched_terms_v2"] = sorted(matched_terms)[:50]
                article["title_matches_v2"] = details["title_matches"]
                article["first_matches_v2"] = details["first_matches"]
                article["rest_matches_v2"] = details["rest_matches"]
                fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                kept_articles += 1

            if total_articles % 100000 == 0:
                print(f"Processed {total_articles:,} articles... kept {kept_articles:,}")

    print(f"Total articles seen: {total_articles:,}")
    print(f"Kept CS articles v2: {kept_articles:,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()