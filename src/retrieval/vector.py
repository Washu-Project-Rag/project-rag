from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_loc = "data/chroma_db"

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
    
# Connect LLM with vector store
vector_store = initialize_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
