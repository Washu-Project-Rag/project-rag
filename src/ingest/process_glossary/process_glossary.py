import json
import re
from pathlib import Path
from nltk.stem import PorterStemmer

# ==========
# Paths
# ==========
BASE_DIR = Path("/Users/hxxy/Desktop/WashU/2026spring/ESE 5971 - Practicum in Data Analytics & Statistics/project/project-rag/src/ingest")
GLOSSARY_FILE = BASE_DIR / "glossary_terms.json"
OUTPUT_FILE = BASE_DIR / "processed_glossary_terms.json"

# ==========
# Token tools
# ==========
stemmer = PorterStemmer()

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())

def normalize_tokens(tokens: list[str]) -> list[str]:
    return [stemmer.stem(tok) for tok in tokens]

def preprocess_text(text: str) -> list[str]:
    return normalize_tokens(tokenize(text))

def load_glossary_terms(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t.strip().lower() for t in data["terms"] if t.strip()]

def prepare_glossary_terms(terms: list[str]):
    prepared = []
    for term in terms:
        term_tokens = preprocess_text(term)
        if term_tokens:
            prepared.append({
                "raw_term": term,
                "processed_tokens": term_tokens
            })
    return prepared

# ==========
# Main
# ==========
glossary_terms = load_glossary_terms(GLOSSARY_FILE)
prepared_terms = prepare_glossary_terms(glossary_terms)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(prepared_terms, f, ensure_ascii=False, indent=2)

print(f"Saved processed glossary to: {OUTPUT_FILE}")
print("First 20 processed terms:")
for item in prepared_terms[:20]:
    print(item)