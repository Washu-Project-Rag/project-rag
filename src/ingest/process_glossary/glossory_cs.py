import json
import re
import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Glossary_of_computer_science"
OUTPUT_JSON = "src/ingest/glossary_terms.json"


def clean_term(term: str) -> str:
    term = re.sub(r"\[[^\]]*\]", "", term)   # remove bracket refs
    term = re.sub(r"\s+", " ", term).strip()
    return term


def is_valid_term(term: str) -> bool:
    """
    Filter out obvious navigation / junk / non-term text.
    """
    if not term:
        return False
    if len(term) < 2 or len(term) > 120:
        return False

    lower = term.lower()

    bad_exact = {
        "edit",
        "computer science",
        "history",
        "outline",
        "glossary",
        "category",
        "see also",
        "references",
        "contents",
        "main page",
        "current events",
        "random article",
        "about wikipedia",
        "help",
        "community portal",
        "recent changes",
        "upload file",
        "special pages",
        "read",
        "view history",
        "download as pdf",
        "printable version",
    }
    if lower in bad_exact:
        return False

    # skip single letters / section labels
    if re.fullmatch(r"[A-Z]", term):
        return False

    # skip very noisy phrases
    bad_contains = [
        "jump to content",
        "free encyclopedia",
        "wikidata",
        "donate",
        "log in",
        "create account",
    ]
    if any(x in lower for x in bad_contains):
        return False

    return True


def fetch_glossary_terms(url: str) -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        raise RuntimeError("Could not find mw-content-text block.")

    terms = []

    # The glossary entries appear as bolded linked terms at the start of definitions.
    # On this page, many terms are inside <dt> ... <a>TERM</a>
    for dt in content.find_all("dt"):
        a = dt.find("a")
        if a:
            term = clean_term(a.get_text(" ", strip=True))
            if is_valid_term(term):
                terms.append(term)

    # Fallback: if dt parsing misses some items, also collect bold anchor text
    if len(terms) < 50:
        for b in content.find_all("b"):
            a = b.find("a")
            if a:
                term = clean_term(a.get_text(" ", strip=True))
                if is_valid_term(term):
                    terms.append(term)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


def save_terms_to_json(terms: list[str], output_path: str):
    payload = {
        "source": URL,
        "count": len(terms),
        "terms": terms
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    terms = fetch_glossary_terms(URL)
    save_terms_to_json(terms, OUTPUT_JSON)
    print(f"Saved {len(terms)} terms to {OUTPUT_JSON}")
    print("First 30 terms:")
    for t in terms[:30]:
        print("-", t)