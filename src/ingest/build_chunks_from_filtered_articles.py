import os
import re
import json
import hashlib
from typing import List, Dict, Iterable, Tuple, Optional

from tqdm import tqdm


# -----------------------
# Heading / structure parsing
# -----------------------

# Traditional wiki heading format
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

# Heuristic top-level headings commonly seen in WikiExtractor-cleaned text
TOP_LEVEL_HEADINGS = {
    "History", "Overview", "Background", "Career", "Club career", "Professional career",
    "Academic career", "Political career", "Military career", "Business career",
    "International career", "Life", "Personal life", "Early life", "Education",
    "Outside football", "Post-retirement", "Legacy", "Reception", "Style of play",
    "Works", "Bibliography", "Discography", "Filmography", "Honours", "Honors",
    "Awards", "See also", "References", "External links", "Publications",
    "Gameplay", "Development", "Plot", "Premise", "Applications", "Examples",
    "Representation", "Analysis", "Properties", "Cryptographic analysis",
    "Design", "Operation", "Architecture", "Implementation"
}

# Strong signals that a short standalone line is probably NOT a heading
SENTENCE_LIKE_RE = re.compile(r"[,:;!?]")


def stable_chunk_id(page_id: str, title: str, section_path: List[str], para_idx: int, sub_idx: int) -> str:
    """
    Stable deterministic chunk ID.
    """
    base = f"{page_id}||{title}||{' > '.join(section_path)}||{para_idx}||{sub_idx}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"wiki_{h}"


def iter_articles_from_jsonl(path: str) -> Iterable[Dict]:
    """
    Read already-filtered article JSONL.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_heading(line: str) -> str:
    """
    Normalize heading text:
    - strip whitespace
    - remove trailing period
    """
    line = line.strip()
    if line.endswith("."):
        line = line[:-1].strip()
    return line


def is_short_heading_line(line: str) -> bool:
    """
    Heuristic for WikiExtractor-cleaned heading lines like:
      Club career.
      Sporting.
      Barcelona.
      International career.
      Personal life.
    """
    s = line.strip()
    if not s:
        return False

    if len(s) > 80:
        return False

    if len(s.split()) > 8:
        return False

    if SENTENCE_LIKE_RE.search(s):
        return False

    if s.endswith("."):
        return True

    if len(s.split()) <= 4 and s[0].isupper():
        return True

    return False


def infer_heading_level(heading: str, current_top: Optional[str]) -> int:
    """
    Infer heading level from cleaned text.

    Returns:
      1 -> top-level section
      2 -> second-level subsection
    """
    if heading in TOP_LEVEL_HEADINGS:
        return 1

    if current_top is not None:
        return 2

    return 1


def split_long_paragraph(para: str, max_chars: int = 1200) -> List[str]:
    """
    Character-based split for long paragraphs.
    """
    para = para.strip()
    if len(para) <= max_chars:
        return [para]

    chunks = []
    start = 0
    while start < len(para):
        end = min(len(para), start + max_chars)

        cut = max(
            para.rfind(". ", start, end),
            para.rfind("\n", start, end)
        )

        if cut <= start + 200:
            cut = end
        else:
            cut += 1

        chunk = para[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut

    return chunks


def parse_sections_and_paragraphs(text: str) -> Iterable[Tuple[List[str], int, str]]:
    """
    Parse article text into:
      (section_path, paragraph_index, paragraph_text)

    Supports BOTH:
    1) explicit wiki headings, e.g. == Heading ==
    2) WikiExtractor-cleaned short standalone heading lines
    """
    lines = (text or "").splitlines()

    current_top: Optional[str] = None
    current_section_path: List[str] = []
    paragraph_index = 0

    def emit_paragraph(section_path: List[str], paragraph_text: str):
        nonlocal paragraph_index
        para = paragraph_text.strip()
        if para and len(para) >= 40:
            yield (section_path.copy(), paragraph_index, para)
            paragraph_index += 1

    for raw_ln in lines:
        ln = raw_ln.strip()
        if not ln:
            continue

        # Case 1: explicit wiki heading == Heading ==
        m = HEADING_RE.match(ln)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()

            if level <= 2:
                current_top = heading
                current_section_path = [heading]
            else:
                if current_top is None:
                    current_top = heading
                    current_section_path = [heading]
                else:
                    current_section_path = [current_top, heading]
            continue

        # Case 2: cleaned heading line like "Club career."
        if is_short_heading_line(ln):
            heading = normalize_heading(ln)
            level = infer_heading_level(heading, current_top)

            if level == 1:
                current_top = heading
                current_section_path = [heading]
            else:
                if current_top is None:
                    current_top = heading
                    current_section_path = [heading]
                else:
                    current_section_path = [current_top, heading]
            continue

        # Case 3: normal paragraph line
        yield from emit_paragraph(current_section_path, ln)


def main(
    input_path: str = "data/processed/wiki_cs_articles_v3.jsonl",
    output_path: str = "data/processed/corpus_chunks_filtered.jsonl",
    source_name: str = "enwiki_pages_articles_latest",
    max_articles: Optional[int] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    kept_articles = 0
    kept_chunks = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for art in tqdm(iter_articles_from_jsonl(input_path), desc="articles"):
            page_id = str(art.get("id", "") or "")
            revid = str(art.get("revid", "") or "")
            url = art.get("url", "") or ""
            title = art.get("title", "") or ""
            text = art.get("text", "") or ""

            kept_articles += 1

            for section_path, para_idx, para in parse_sections_and_paragraphs(text):
                if not section_path:
                    section_path = ["(root)"]

                sub_chunks = split_long_paragraph(para, max_chars=1200)

                for sub_idx, sub in enumerate(sub_chunks):
                    chunk_id = stable_chunk_id(page_id, title, section_path, para_idx, sub_idx)

                    rec = {
                        "chunk_id": chunk_id,
                        "page_id": page_id,
                        "title": title,
                        "section_path": section_path,
                        "paragraph_index": para_idx,
                        "text": sub,
                        "source": source_name,
                        "url": url,
                        "revid": revid,
                    }

                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept_chunks += 1

            if max_articles is not None and kept_articles >= max_articles:
                break

    print(f"Done. kept_articles={kept_articles}, kept_chunks={kept_chunks}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()