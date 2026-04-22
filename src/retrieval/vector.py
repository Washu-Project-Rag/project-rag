from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_loc = "data/chroma_db"


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


def get_vector_store():
    return Chroma(
        collection_name="wikipedia_cs",
        persist_directory=db_loc,
        embedding_function=embeddings,
    )


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
    
vector_store = get_vector_store()
build_vector_store_if_needed(vector_store)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 100}
)

if __name__ == "__main__":
    print("DB PATH EXISTS:", os.path.exists(db_loc))
    print("DB FILES:", os.listdir(db_loc) if os.path.exists(db_loc) else None)

    build_vector_store_if_needed(vector_store)
