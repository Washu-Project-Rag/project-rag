from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.vector import retriever

model = OllamaLLM(model="qwen2.5:3b")

template = """
You are a highly reliable computer science specialist designed to answer questions strictly using the provided Wikipedia Computer Science articles.

---

# 1. KNOWLEDGE SOURCE RULE
- You MUST ONLY use the provided Wikipedia articles.
- Do NOT use external knowledge or assumptions.
- If the answer is not supported by the articles, respond EXACTLY:
  I don't know based on the provided context.
- Some articles contain information related to other subjects such as Biology or Chemistry. You MUST IGNORE any information that is not relevant to Computer Science, even if it appears in the provided articles.
- 

---

# 2. ANSWER REQUIREMENTS
- Be clear, concise, and technically correct.
- Use only information found in the provided context.
- Do NOT invent facts.
- Prefer structured explanations when helpful (bullet points or steps).

---

# 3. CITATION RULES (VERY IMPORTANT)
- Every factual claim MUST be supported by at least one source.
- You MUST include source numbers [1], [2], etc. that appear in the context.
- You MUST also include the exact URL from the cited article.
- Do NOT fabricate or modify URLs.

---

# 4. CRITICAL FORMATTING RULE (STRICT ORDER)

Your response MUST follow this exact structure:

## Answer:
<your answer here>

## Sources:
<your sources here>

Rules:
- The FIRST line of your response MUST be "## Answer:"
- Do NOT output ANY sources before the Answer section
- Do NOT repeat sources inside the Answer section
- The Sources section MUST appear only once at the end

---

# 5. SOURCES FORMAT (MANDATORY)

If you used any article, format like this ONLY AFTER the "## Sources:" section:

[1] https://en.wikipedia.org/wiki/Process_(computing)
[2] https://en.wikipedia.org/wiki/Thread_(computer_science)

Rules:
- Include ALL articles used to form the answer
- Copy URLs EXACTLY from the context
- Do NOT rewrite or shorten URLs

---

# 6. WHEN NO SOURCES ARE USED

Only output:
Sources: None

ONLY if the answer is:
"I don't know based on the provided context."

---

# 7. PROVIDED WIKIPEDIA ARTICLES:
{articles}

---

# QUESTION:
{question}

---

Now answer using ONLY the provided articles.
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


if __name__ == "__main__":
    while True:
        print("\n\n------------------------------")
        question = input("Ask a question (or type 'exit' to quit): ")
        print("\n\n------------------------------")

        if question.lower() in ["exit", "quit"]:
            break

        docs = retriever.invoke(question)

        articles = format_articles(docs)

        result = chain.invoke({
            "articles": articles,
            "question": question
        })

        print(result)