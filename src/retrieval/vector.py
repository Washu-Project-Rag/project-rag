from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_json("data/processed/corpus_chunks.jsonl", lines=True)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_loc = "data/chroma_db"
add_documents = not os.path.exists(db_loc)

if add_documents:
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

# Initialize vector store
vector_store = Chroma(
    collection_name="wikipedia_cs",
    persist_directory=db_loc,
    embedding_function=embeddings,
)

# Embedding step: only add documents if the vector store is not already built
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
# Connect LLM with vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
