from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.vector import get_retriever

model = OllamaLLM(model="qwen2.5:3b")

template = """
You are a highly reliable computer science specialist designed to answer questions strictly using the provided Wikipedia Computer Science articles.

# RULES
- You MUST ONLY use the provided Wikipedia articles.
- Do NOT use external knowledge or assumptions.
- If the answer is not supported by the articles, respond EXACTLY:

  I don't know based on the provided context.
  
  ## Sources: 
  None
  
- Some articles contain information related to other subjects such as Biology or Chemistry. You MUST IGNORE any information that is not relevant to Computer Science, even if it appears in the provided articles.
- Any source used outside of the prrovided articles is considered a hallucination and will be penalized.
- Answer in one concise paragraph.
- Do not use any bullet points, lists, or markdown formatting. 

# PROVIDED WIKIPEDIA ARTICLES:
{articles}

# QUESTION:
{question}

# OUTPUT FORMAT (STRICT ORDER)

Return exactly the following format, with no deviations:

## Answer:
<answer paragraph>

## Sources:
[1] <URL of source 1>
[2] <URL of source 2>
and so on, if more sources are used.

# SOURCES RULES (VERY STRICT)

- Only include URLs that are explicitly used in the answer.
- Each URL must appear ONLY ONCE.
- Number sources sequentially: [1], [2], [3], ...
- Do not include the word "None" if any URL is listed.
- If no sources were used, write exactly:

## Sources:
None

- Otherwise, do NOT include the "None" line at all.

- Do not output anything after the Sources section.
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_articles(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        title = metadata.get("title", "Unknown")
        url = metadata.get("url", "Unknown")

        formatted.append(
            f"[{i}] {title} | {url}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


def generate_answer(question, k=15):
    retriever = get_retriever(k)
    docs = retriever.invoke(question)
    articles = format_articles(docs)

    result = chain.invoke({
        "articles": articles,
        "question": question
    })

    return result


if __name__ == "__main__":
    retriever = get_retriever(k=15)
    
    while True:
        print("\n\n----------------------------------------------------------------------------------------------------------------")
        question = input("Ask a question (or type 'exit' or 'quit' to quit): ")
        print("\n\n----------------------------------------------------------------------------------------------------------------")

        if question.lower() in ["exit", "quit"]:
            break

        result = generate_answer(question, k=15)
        print(result)