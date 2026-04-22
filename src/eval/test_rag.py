import pandas as pd
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from src.generation.answer import chain
from src.retrieval.vector import vector_store


queries = pd.read_json("eval/queries.jsonl", lines=True)
sample_queries = list(queries["question"])
# print(sample_queries)

answers = pd.read_json("eval/answers.jsonl", lines=True)
expected_answers = list(answers["answer"])

TOP_K = [5,10,15,20]
all_rows = []

for k in TOP_K:
    print(f"-------------- Running k = {k} --------------")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 100}
    )

    # DeepEval Implementation
    correctness_metric = GEval(
        name="Correctness",
        model="gpt-4o-mini",
        criteria=(
            "You are evaluating ONLY semantic correctness.\n\n"

            "The answer is correct if it conveys the same meaning as the expected answer.\n\n"

            "STRICT RULES:\n"
            "- Ignore all formatting differences (bullets, numbering, markdown, headings, spacing).\n"
            "- Ignore structure differences (single paragraph vs multiple bullets).\n"
            "- Ignore citation styles and section labels like 'Answer' or 'Sources'.\n"
            "- Focus ONLY on whether the same computer science concepts are correctly explained.\n\n"

            "FOCUS ON:\n"
            "- Correctness of definitions\n"
            "- Accuracy of relationships between concepts\n"
            "- Presence of key information\n"
        ),
        evaluation_steps=[
            "Extract semantic meaning from both actual and expected answers.",
            "Treat all formatting styles as equivalent.",
            "Compare only factual correctness and completeness of CS concepts.",
            "Ignore markdown, bullet points, and structural differences.",
            "Penalize missing or incorrect technical content only."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
    )

    faithfulness_metric = GEval(
        name="Faithfulness",
        model="gpt-4o-mini",
        criteria=(
            "You are evaluating whether the answer is fully derived from the retrieved context.\n\n"

            "STRICT RULES:\n"
            "- Ignore formatting differences completely.\n"
            "- Ignore headings, bullet points, and answer structure.\n"
            "- Ignore citation formatting.\n\n"

            "FOCUS ONLY ON FACTUAL GROUNDING:\n"
            "- Every claim in the answer must be supported by retrieved context.\n"
            "- No external or hallucinated information is allowed.\n"
            "- Paraphrasing is allowed if meaning stays consistent with the context.\n"
        ),
        evaluation_steps=[
            "Extract atomic factual claims from the answer.",
            "Check whether each claim is supported by the retrieved context.",
            "Ignore formatting and structure completely.",
            "Mark hallucinated or unsupported claims as incorrect.",
            "Reward answers fully grounded in context."
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
    )

    context_precision_recall_metric = GEval(
        name="Context Precision Recall",
        model="gpt-4o-mini",
        criteria=(
            "You are evaluating whether the retrieved context is relevant and sufficient.\n\n"

            "STRICT RULES:\n"
            "- Ignore formatting differences in both answers and context.\n"
            "- Ignore markdown, bullets, or structure differences.\n\n"

            "FOCUS ONLY ON CONTENT QUALITY:\n"
            "- Precision: are retrieved chunks relevant to the question?\n"
            "- Recall: does the context contain enough information to answer the question?\n"
            "- Ignore how the answer is formatted or structured.\n"
        ),
        evaluation_steps=[
            "Analyze whether retrieved documents are relevant to the question.",
            "Check whether important information is missing from retrieval.",
            "Ignore formatting and presentation differences.",
            "Evaluate only semantic relevance and completeness.",
            "Penalize irrelevant or missing context."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
    )

    # test_cases = []

    for question, expected in zip(sample_queries, expected_answers):
        docs = retriever.invoke(question)
        retrieval_context = [doc.page_content for doc in docs]

        # Generate answer using RAG chain
        actual_output = chain.invoke({
            "articles": docs,
            "question": question
        })

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context
        )
        
        correctness_metric.measure(test_case)
        faithfulness_metric.measure(test_case)
        context_precision_recall_metric.measure(test_case)

        all_rows.append({
            "k": k,
            "question": question,
            "correctness_score": correctness_metric.score,
            "correctness_pass": correctness_metric.score >= 0.5,
            "faithfulness_score": faithfulness_metric.score,
            "faithfulness_pass": faithfulness_metric.score >= 0.5,
            "context_precision_recall_score": context_precision_recall_metric.score,
            "context_pass": context_precision_recall_metric.score >= 0.5
        })
        
        # test_cases.append(test_case)
        
df = pd.DataFrame(all_rows)
df.to_csv("eval/eval_scores.csv", index=False)

# Run Evaluation
# evaluate(test_cases=test_cases, metrics=[correctness_metric])
# evaluate(test_cases=test_cases, metrics=[faithfulness_metric])
# evaluate(test_cases=test_cases, metrics=[context_precision_recall_metric])