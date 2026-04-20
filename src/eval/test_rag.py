import pandas as pd
# from deepeval.metrics import (
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
#     ContextualRelevancyMetric
# )
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
# from ragas import EvaluationDataset


queries = pd.read_json("eval/queries.jsonl", lines=True)
sample_queries = list(queries["question"])
# print(sample_queries)

answers = pd.read_json("eval/expected_answers.jsonl", lines=True)
expected_answers = list(answers["answer"])

###########################
# RAGAS Implementation 
###########################

# dataset = []
# for query,reference in zip(sample_queries,expected_answers):

#     relevant_docs = rag.get_most_relevant_docs(query)
#     response = rag.generate_answer(query, relevant_docs)
#     dataset.append(
#         {
#             "user_input":query,
#             "retrieved_contexts":relevant_docs,
#             "response":response,
#             "reference":reference
#         }
#     )

###########################
# DeepEval Implementation 
###########################

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

faithfulness = GEval(
    name="Faithfulness",
    evaluation_steps=[
        "Extract medical claims or diagnoses from the actual output.",
        "Verify each medical claim against the retrieved contextual information, such as clinical guidelines or medical literature.",
        "Identify any contradictions or unsupported medical claims that could lead to misdiagnosis.",
        "Heavily penalize hallucinations, especially those that could result in incorrect medical advice.",
        "Provide reasons for the faithfulness score, emphasizing the importance of clinical accuracy and patient safety."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="It depends, some might consider the cat, while others might argue the dog.",
    expected_output="The cat."
)

# To run metric as a standalone
# correctness_metric.measure(test_case)
# print(correctness_metric.score, correctness_metric.reason)

evaluate(test_cases=[test_case], metrics=[correctness_metric])