import os
from functools import partial


# Get extra message text
curr_dir = os.path.split(__file__)[0]
extra_text_path = os.path.join(curr_dir, "..", "extra_text.md")
with open(extra_text_path, "r") as f:
    EXTRA_MESSAGE = f.read()

def _doc_to_text(example, formatting_fn):
    return EXTRA_MESSAGE + "\n" + formatting_fn(example)

def format_regular(example):
    return f"Question: {example['question']}\nAnswer:"

doc_to_text = partial(_doc_to_text, formatting_fn=format_regular)