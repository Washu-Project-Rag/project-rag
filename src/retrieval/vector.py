from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_loc = "data/chroma_db"

_embeddings = None
_vector_store = None
_retriever_cache = {}

def load_documents():
    df = pd.read_json("data/processed/corpus_chunks_filtered.jsonl", lines=True)
    documents = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=(
                f"Title: {row['title']}\n"
                f"Section: {' | '.join(row['section_path'])}\n"
                f"Content: {row['text']}"
            ),
            metadata={"url": row["url"]},
            id=str(i)
        )
        documents.append(doc)
        ids.append(str(i))

    return documents, ids


def build_vector_store_if_needed(vector_store):
    try:
        count = vector_store._collection.count()
    except Exception:
        count = 0

    # print("COLLECTION COUNT:", count)

    if count > 0:
        # print("DB already built — skipping indexing.")
        return

    documents, ids = load_documents()
    # print("LOADED DOCUMENTS:", len(documents))

    batch_size = 5000

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        # print(f"ADDING BATCH {i} to {i + len(batch_docs)}")

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )

    # print("DONE INDEXING")
    
    
def get_embeddings():
    global _embeddings
    
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return _embeddings
    

def get_vector_store():
    global _vector_store
    
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name="wikipedia_cs",
            persist_directory=db_loc,
            embedding_function=get_embeddings(),
        )
        
        build_vector_store_if_needed(_vector_store)
        
    return _vector_store

# def get_vector_store():
#     return Chroma(
#         collection_name="wikipedia_cs",
#         persist_directory=db_loc,
#         embedding_function=embeddings,
#     )


def get_retriever(k=15):
    global _retriever_cache
    
    if k not in _retriever_cache:
        store = get_vector_store()
        _retriever_cache[k] = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 100}
        )

    return _retriever_cache[k]

    
# vector_store = get_vector_store()
# build_vector_store_if_needed(vector_store)

# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 15, "fetch_k": 100}
# )

if __name__ == "__main__":
    store = get_vector_store()
    
    print("DB PATH EXISTS:", os.path.exists(db_loc))
    print("DB FILES:", os.listdir(db_loc) if os.path.exists(db_loc) else None)
    print("DB COUNT:", store._collection.count())
