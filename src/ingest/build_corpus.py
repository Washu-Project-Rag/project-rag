import os
import re
import json
import glob
import hashlib
from typing import List, Dict, Iterable, Tuple, Optional

from tqdm import tqdm


# -----------------------
# 1) Simple technical/engineering topic filter
# -----------------------
TECH_KEYWORDS = [
    # general CS/engineering
    "algorithm", "data structure", "computer", "computing", "software", "hardware",
    "engineering", "electrical", "mechanical", "civil", "chemical", "aerospace",
    # systems
    "operating system", "kernel", "linux", "windows", "unix", "compiler", "runtime",
    "distributed", "cloud", "virtualization", "container", "kubernetes", "docker",
    # networking
    "network", "protocol", "tcp", "udp", "ip", "http", "tls", "ssl", "dns", "routing",
    # security
    "encryption", "cryptography", "cipher", "authentication", "authorization", "malware",
    # data
    "database", "sql", "index", "query", "transaction", "replication",
    # ML
    "machine learning", "neural network", "deep learning", "training", "inference",
    # programming
    "programming", "python", "java", "c++", "javascript", "api",
]

TECH_REGEX = re.compile(r"\b(" + "|".join(re.escape(k) for k in TECH_KEYWORDS) + r")\b", re.IGNORECASE)
NEGATIVE_KEYWORDS = [
    "railway station", "football", "basketball", "album", "song",
    "film", "actress", "actor", "politician", "school", "village",
    "river", "mountain", "species", "genus", "taxon",

    # politics / government
    "agreement", "treaty", "constitution", "election", "political party",
    "parliament", "government", "president", "prime minister",

    # entertainment / media
    "tv network", "television", "channel", "episode", "series",
    "band", "music", "record label", "filmography",
]

NEG_REGEX = re.compile(r"\b(" + "|".join(re.escape(k) for k in NEGATIVE_KEYWORDS) + r")\b", re.IGNORECASE)

# Common heading format in WikiExtractor output: == Section ==, === Subsection ===
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def is_technical_article(title: str, text: str, min_hits: int = 4) -> bool:
    """
    Stricter keyword filter:
    - Reject obvious non-technical topics with a negative keyword list
    - Add a title-level guardrail for common non-technical categories
    - Require more technical keyword hits
    """
    title_l = (title or "").lower()

    # Fast title guardrail (catches cases like "Taif Agreement", "Boomerang (TV network)")
    if any(x in title_l for x in ["agreement", "treaty", "tv network", "television", "album", "song", "film"]):
        return False

    blob = (title or "") + "\n" + (text or "")[:4000]

    # Negative filter
    if NEG_REGEX.search(blob):
        return False

    # Technical keyword hits threshold
    hits = len(TECH_REGEX.findall(blob))
    return hits >= min_hits


def stable_chunk_id(title: str, section_path: List[str], para_idx: int, sub_idx: int) -> str:
    """Generate a stable, deterministic chunk id based on content coordinates."""
    base = f"{title}||{' > '.join(section_path)}||{para_idx}||{sub_idx}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"wiki_{h}"


def iter_articles_from_wikiextractor_files(paths: List[str]) -> Iterable[Dict]:
    """
    WikiExtractor with --json outputs one JSON object per line.
    Typical fields: id, title, text.
    """
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue


def parse_sections_and_paragraphs(text: str) -> Iterable[Tuple[List[str], int, str]]:
    """
    Parse an article into (section_path, paragraph_index, paragraph_text).
    - Maintain a section stack using headings like '== Heading =='
    - Split paragraphs by blank lines
    """
    section_stack: List[str] = []
    paragraph_index = 0

    # Iterate line-by-line; headings update section_stack, other lines go into the current paragraph buffer
    buf: List[str] = []

    def flush_buf():
        nonlocal paragraph_index, buf
        para = "\n".join(buf).strip()
        buf = []
        if para:
            # Drop very short paragraphs
            if len(para) >= 120:
                yield (section_stack.copy(), paragraph_index, para)
                paragraph_index += 1

    lines = (text or "").splitlines()
    for ln in lines:
        m = HEADING_RE.match(ln.strip())
        if m:
            # Flush any buffered paragraph before entering a new section
            yield from flush_buf()

            level = len(m.group(1))  # '==' is 2, '===' is 3, etc.
            heading = m.group(2).strip()

            # level=2 means top-level section; deeper levels increase nesting
            # stack depth = level - 2
            target_depth = max(0, level - 2)
            section_stack = section_stack[:target_depth]
            section_stack.append(heading)
        else:
            # Treat blank lines as paragraph boundaries
            if ln.strip() == "":
                yield from flush_buf()
            else:
                buf.append(ln)

    # Flush the last paragraph
    yield from flush_buf()


def split_long_paragraph(para: str, max_chars: int = 1800) -> List[str]:
    """
    A simple character-length splitter to prevent extremely long paragraphs.
    You can later replace this with token-based splitting.
    """
    para = para.strip()
    if len(para) <= max_chars:
        return [para]

    chunks = []
    start = 0
    while start < len(para):
        end = min(len(para), start + max_chars)
        # Prefer splitting at sentence boundaries or newlines
        cut = max(para.rfind(". ", start, end), para.rfind("\n", start, end))
        if cut <= start + 200:  # If no good cut point is found, hard cut
            cut = end
        chunks.append(para[start:cut].strip())
        start = cut
    return [c for c in chunks if c]


def main(
    input_glob: str = "data/raw/wiki_extracted_sample/*",
    output_path: str = "data/processed/corpus_chunks.jsonl",
    corpus_version: str = "enwiki_pages_articles_snapshot",
    min_hits: int = 4,
    max_articles: Optional[int] = 2000,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    paths = sorted(glob.glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    out_f = open(output_path, "w", encoding="utf-8")

    kept_articles = 0
    kept_chunks = 0

    for art in tqdm(iter_articles_from_wikiextractor_files(paths), desc="articles"):
        title = art.get("title", "") or ""
        text = art.get("text", "") or ""

        if not is_technical_article(title, text, min_hits=min_hits):
            continue

        kept_articles += 1

        for section_path, para_idx, para in parse_sections_and_paragraphs(text):
            # If the section stack is empty, assign a default root marker (optional)
            if not section_path:
                section_path = ["(root)"]

            sub_chunks = split_long_paragraph(para, max_chars=1800)
            for sub_idx, sub in enumerate(sub_chunks):
                chunk_id = stable_chunk_id(title, section_path, para_idx, sub_idx)
                rec = {
                    "chunk_id": chunk_id,
                    "title": title,
                    "section_path": section_path,
                    "paragraph_index": para_idx,
                    "text": sub,
                    "source": corpus_version,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept_chunks += 1

        if max_articles is not None and kept_articles >= max_articles:
            break

    out_f.close()
    print(f"Done. kept_articles={kept_articles}, kept_chunks={kept_chunks}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()