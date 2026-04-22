from src.retrieval.vector import retriever

query = "process vs thread"

docs = retriever.invoke(query)

print("NUM DOCS:", len(docs))

for i, d in enumerate(docs[:3]):
    print("\n--- DOC", i, "---")
    print(d.page_content[:500])