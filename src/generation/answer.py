from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.vector import retriever

model = OllamaLLM(model="qwen2.5:3b")

template = """
You are an assistant which answers questions based on the Wikipedia Computer Science 
articles provided to you. Use only this knowledge to answer the quesiton, 
not your own internal knowledge. If you don't know the answer, say you don't know. 

The source article(s): {articles}

The question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------------")
    question = input("Ask a question (or type 'exit' to quit): ")
    print("\n\n------------------------------")
    if question.lower() in ["exit", "quit"]:
        break
    
    articles = retriever.invoke(question)
    result = chain.invoke({"articles": articles, "question": question})
    print(result)