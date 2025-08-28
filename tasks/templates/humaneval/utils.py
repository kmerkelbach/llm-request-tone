import evaluate as hf_evaluate
import os
from functools import partial


# Get extra message text
curr_dir = os.path.split(__file__)[0]
extra_text_path = os.path.join(curr_dir, "extra_text.txt")
with open(extra_text_path, "r") as f:
    EXTRA_MESSAGE = f.read()


def _doc_to_text(example, formatting_fn):
    return EXTRA_MESSAGE + "\n" + formatting_fn(example)


def format_regular(example):
    return f"{example['prompt']}"


def format_instruct(example):
    return (f"Write a solution to the following problem and make sure that it passes the"
            f" tests:\n```python\n{example['prompt']}\n```\n")


doc_to_text = partial(_doc_to_text, formatting_fn=format_regular)
doc_to_text_instruct = partial(_doc_to_text, formatting_fn=format_instruct)


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
