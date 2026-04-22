import json
from pathlib import Path

INPUT_FILE = Path("src/ingest/glossary_terms.json")
OUTPUT_FILE = Path("src/ingest/glossary_terms_grouped.json")

CORE_TERMS = {
    "algorithm", "algorithm design", "algorithmic efficiency",
    "application programming interface", "application software",
    "artificial intelligence", "ascii", "automata theory",
    "bandwidth", "binary number", "binary search algorithm",
    "bioinformatics", "bit rate", "booting", "boolean algebra",
    "callback", "central processing unit", "cipher", "ci/cd",
    "cloud computing", "coding theory", "compiler",
    "computability theory", "computational biology",
    "computational chemistry", "computational complexity theory",
    "computational model", "computational neuroscience",
    "computational physics", "computational science",
    "computer architecture", "computer data storage",
    "computer ethics", "computer graphics", "computer network",
    "computer program", "computer programming", "computer science",
    "computer scientist", "computer security", "computer vision",
    "computing", "concurrency", "control flow", "cryptography",
    "csv", "cyberspace", "data center", "data mining",
    "data science", "data structure", "data type", "database",
    "daemon", "debugging", "digital data", "digital signal processing",
    "distributed computing", "dns", "domain name system",
    "edge device", "emulator", "encryption", "event-driven programming",
    "exception handling", "fault-tolerant computer system",
    "filename extension", "filter (software)", "floating-point arithmetic",
    "for loop", "formal methods", "formal verification",
    "functional programming", "hash function", "hash table",
    "heapsort", "human-computer interaction", "image processing",
    "imperative programming", "information visualization",
    "information retrieval", "input/output",
    "integrated development environment", "intelligent agent",
    "internet bot", "library (computing)", "linked list", "linker",
    "logic programming", "machine learning", "machine vision",
    "mathematical logic", "modem", "natural language processing",
    "numerical analysis", "numerical method", "object code",
    "object-oriented programming", "open-source software",
    "operating system", "optical fiber", "parallel computing",
    "pointer", "priority queue", "procedural programming",
    "programming language", "programming language implementation",
    "programming language theory", "prolog", "python",
    "quantum computing", "queue", "quicksort",
    "r programming language", "radix", "recursion",
    "reference counting", "regression testing", "relational database",
    "robotics", "router", "routing table", "run time",
    "run time error", "search algorithm", "selection sort",
    "serializability", "serialization", "software", "software agent",
    "software construction", "software deployment", "software design",
    "software development", "software engineering",
    "software maintenance", "software prototyping",
    "software requirements specification", "software testing",
    "sorting algorithm", "source code", "stack", "subroutine",
    "syntax", "syntax error", "system console",
    "technical documentation", "type theory",
    "uniform resource locator", "user agent", "user interface",
    "user interface design", "virtual machine", "web crawler",
    "wi-fi", "xhtml"
}

RELATED_TERMS = {
    "abstraction", "abstract data type", "abstract method",
    "agile software development", "agent architecture",
    "agent-based model", "aggregate function", "artifact",
    "associative array", "automated reasoning", "benchmark",
    "big data", "big o notation", "blacklist", "byte",
    "closure", "coding", "comma-separated values", "computer",
    "continuous delivery", "continuous deployment",
    "continuous integration", "creative commons", "declaration",
    "disk storage", "documentation", "download",
    "evolutionary computing", "feasibility study", "game theory",
    "garbage in, garbage out", "gigabyte", "global variable",
    "graph theory", "heap", "ide", "incremental build model",
    "insertion sort", "instruction cycle", "integration testing",
    "intellectual property", "interface", "internal documentation",
    "internet", "iteration", "java", "kernel", "linear search",
    "loader", "matrix", "merge sort", "methodology", "node",
    "number theory", "pair programming", "peripheral",
    "postcondition", "precondition", "primary storage",
    "procedural generation", "program lifecycle phase",
    "reliability engineering", "requirements analysis",
    "secondary storage", "semantics", "service level agreement",
    "spiral model", "storage", "structured storage",
    "symbolic computation", "upload", "v-model", "waterfall model",
    "waveform audio file format"
}

WEAK_TERMS = {
    "american standard code for information interchange",
    "array data structure", "assertion", "bayesian programming",
    "best, worst and average case", "bit", "bmp file format",
    "boolean data type", "boolean expression", "character", "class",
    "class-based programming", "client", "cleanroom software engineering",
    "collection", "computation", "computational steering",
    "conditional", "container", "continuation-passing style",
    "cyberbullying", "data", "discrete event simulation", "domain",
    "double-precision floating-point format", "event", "executable",
    "execution", "expression", "field", "graphics interchange format",
    "handle", "hard problem", "identifier", "inheritance",
    "information space analysis", "integer", "invariant", "list",
    "logic error", "memory", "method", "object",
    "object-oriented analysis and design", "parameter",
    "primitive data type", "procedure", "record", "reference",
    "round-off error", "sequence", "server", "set", "state",
    "statement", "stream", "string", "top-down and bottom-up design",
    "tree", "user", "variable"
}

def normalize_term(t: str) -> str:
    return t.strip().lower()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

grouped_terms = []
unassigned = []

for term in data["terms"]:
    norm = normalize_term(term)

    if norm in CORE_TERMS:
        group = "core"
    elif norm in RELATED_TERMS:
        group = "related"
    elif norm in WEAK_TERMS:
        group = "weak"
    else:
        group = "related"   
        unassigned.append(term)

    grouped_terms.append({
        "term": term,
        "group": group
    })

output = {
    "source": data.get("source"),
    "count": len(grouped_terms),
    "group_definitions": {
        "core": "strong evidence of core CS content",
        "related": "related to computing/software/AI but weaker evidence",
        "weak": "too broad/ambiguous; cannot be used alone"
    },
    "terms": grouped_terms
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved grouped glossary to: {OUTPUT_FILE}")
print(f"Unassigned terms defaulted to related: {len(unassigned)}")
print(unassigned[:30])