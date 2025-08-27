import random
import re

import datasets
import os


# Get extra message text
curr_dir = os.path.split(__file__)[0]
extra_text_path = os.path.join(curr_dir, "extra_text.txt")
with open(extra_text_path, "r") as f:
    EXTRA_MESSAGE = f.read()


def doc_to_text(example):
    return EXTRA_MESSAGE + "\n" + f"Question: {example['Question']}\nChoices:\n(A) {example['choice1']}\n(B) {example['choice2']}\n(C) {example['choice3']}\n(D) {example['choice4']}\nAnswer:"


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
