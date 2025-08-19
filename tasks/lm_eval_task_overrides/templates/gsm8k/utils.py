import os
from functools import partial


# Get extra message text
curr_dir = os.path.split(__file__)[0]
extra_text_path = os.path.join(curr_dir, "..", "extra_text.txt")
with open(extra_text_path, "r") as f:
    EXTRA_MESSAGE = f.read()

def _doc_to_text(example, formatting_fn):
    return EXTRA_MESSAGE + "\n" + formatting_fn(example)

def format_regular(example):
    return f"Question: {example['question']}\nAnswer:"

def format_cot(example):
    return f"Q: {example['question']}\n\n    A: "

def format_cot_llama(example):
    return f"Given the following problem, reason and give a final answer to the problem.\nProblem: {example['question']}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"

def format_cot_zeroshot(example):
    return f"Q: {example['question']}\nA: Let's think step by step."

doc_to_text = partial(_doc_to_text, formatting_fn=format_regular)
doc_to_text_cot = partial(_doc_to_text, formatting_fn=format_cot)
doc_to_text_cot_llama = partial(_doc_to_text, formatting_fn=format_cot_llama)
doc_to_text_cot_zeroshot = partial(_doc_to_text, formatting_fn=format_cot_zeroshot)