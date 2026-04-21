from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
import os
import pandas as pd

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_loc = "data/chroma_db"
SIMILARITY_THRESHOLD = 0.75

def load_documents():
    """Loads documents from the processed corpus JSONL file and returns a list of Document objects and their corresponding IDs."""
    df = pd.read_json("data/processed/corpus_chunks_filtered.jsonl", lines=True)
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        doc = Document(
            page_content=(
                row["title"] + " | " + 
                " > ".join(row["section_path"]) + " | " +
                row["text"]),
            metadata={"url": row["url"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(doc)
    return documents, ids

def initialize_vector_store():
    """Initializes the Chroma vector store. If the store already exists, it will be loaded.
    If it does not exist, it will be created and populated with documents."""
    add_documents = not os.path.exists(db_loc)
    vector_store = Chroma(
        collection_name="wikipedia_cs",
        persist_directory=db_loc,
        embedding_function=embeddings,
    )
    
    # Embedding step: only add documents if the vector store is not already built
    if add_documents:
        documents, ids = load_documents()
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store

vector_store = initialize_vector_store()


classifier_model = OllamaLLM(model="qwen2.5:1.5b")

def is_computer_science(doc) -> bool:
    """
    Uses an LLM to determine whether a document is CS-related.
    Returns True/False.
    """

    prompt = f"""
    You are a strict classifier.

    Determine whether the following text is about COMPUTER SCIENCE.

    Computer science includes:
    - algorithms
    - data structures
    - operating systems
    - databases
    - networks
    - programming languages
    - software systems
    - computer architecture
    - artificial intelligence
    - machine learning
    - theoretical CS concepts (complexity, computability, etc.)

    If the text is about biology, medicine, chemistry, physics (non-CS), or general science, answer NO.

    Return ONLY one word: YES or NO.

    TEXT:
    {doc.page_content}
    """

    result = classifier_model.invoke(prompt).strip().upper()

    return "YES" in result


# ----------------------------
# RETRIEVAL WITH FILTERING
# ----------------------------
def retrieve_filtered(query, k=5):
    """
    1. Retrieve candidates via embeddings
    2. Filter by similarity
    3. Filter by CS domain classifier
    """

    results = vector_store.similarity_search_with_score(query, k=k * 10)
    filtered_docs = []

    for doc, score in results:

        # STEP 1: similarity filter (embedding-based)
        if score > (1 - SIMILARITY_THRESHOLD):
            continue

        # STEP 2: semantic domain filter (NO keywords)
        if not is_computer_science(doc):
            continue

        filtered_docs.append(doc)

        if len(filtered_docs) >= k:
            break

    return filtered_docs
    
# Connect LLM with vector store
class RetrieverWrapper:
    def invoke(self, query):
        return retrieve_filtered(query)

retriever = RetrieverWrapper()
