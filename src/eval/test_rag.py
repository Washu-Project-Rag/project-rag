import pandas as pd
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate
from deepeval.metrics import GEval
from src.generation.answer import chain
from src.retrieval.vector import retriever


queries = pd.read_json("eval/queries.jsonl", lines=True)
sample_queries = list(queries["question"])
# print(sample_queries)

answers = pd.read_json("eval/answers.jsonl", lines=True)
expected_answers = list(answers["answer"])

###########################
# DeepEval Implementation 
###########################

correctness_metric = GEval(
    name="Correctness",
    criteria=(
        "The answer must correctly match the expected answer in meaning, "
        "including key computer science concepts, definitions, and relationships. "
        "Paraphrasing is allowed only if the technical meaning is preserved without loss of important details."
    ),
    evaluation_steps=[
        "Compare the chatbot answer to the expected answer.",
        "Allow paraphrasing if technical meaning is preserved.",
        "Penalize incorrect computer science facts.",
        "Penalize missing key technical details.",
        "Penalize contradictions heavily."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

faithfulness_metric = GEval(
    name="Faithfulness",
    criteria=(
        "The answer must be fully supported by the retrieved Wikipedia computer science context. "
        "No external knowledge, assumptions, or hallucinated facts are allowed."
    ),
    evaluation_steps=[
        "Extract factual claims from the chatbot response.",
        "Check whether every claim is supported by the retrieved Wikipedia context.",
        "Claims may include definitions, algorithms, complexity classes, architectures, networking facts, data structures, programming languages, or historical CS facts.",
        "Penalize hallucinated claims heavily.",
        "Penalize contradictions with retrieved context heavily.",
        "Reward answers that stay grounded only in retrieved context."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
)

context_precision_recall_metric = GEval(
    name="Context Precision Recall",
    criteria=(
        "The retrieved context must be both relevant and sufficient to answer the question. "
        "It should include necessary computer science concepts while avoiding irrelevant information."
    ),
    evaluation_steps=[
        "Determine whether retrieved context is relevant to the question.",
        "Precision: penalize irrelevant passages or noisy retrieved chunks.",
        "Recall: penalize if essential information needed for the expected answer is missing.",
        "Reward context that is focused and sufficiently complete.",
        "For technical questions, verify that required definitions, examples, formulas, or algorithm details are present."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT,LLMTestCaseParams.EXPECTED_OUTPUT],
)

test_cases = []

for question, expected in zip(sample_queries, expected_answers):
    # Retrieve docs
    docs = retriever.invoke(question)

    # Convert to plain text context
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

    test_cases.append(test_case)

####################
# Run Evaluation
####################

evaluate(
    test_cases=test_cases,
    metrics=[
        correctness_metric,
        faithfulness_metric,
        context_precision_recall_metric
    ]
)