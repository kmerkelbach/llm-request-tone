import json
import os
import random

from loguru import logger
from typing import List

from .evaluation.lm_eval_shell import run_eval
from .framing.task_framer import TaskFramer
from .framing.dto import ModifiedTask
from .util.utils import get_eval_dir, make_date_string


benchmarks = [
    "mmlu_pro",
    "gpqa_diamond_generative_n_shot",
    "gsm8k",
    "ifeval",
    "truthfulqa_gen",
    "humaneval",
    "mbpp",
    "mbpp_plus"
]


if __name__ == "__main__":
    # Frame tasks
    framer = TaskFramer()
    modified_tasks: List[ModifiedTask] = framer.template_all_tasks()

    # For testing, pick some templated tasks at random
    picked: List[ModifiedTask] = random.sample(modified_tasks, k=3)
    original_tasks = set()
    for task in picked:
        original_tasks.add(task.origin_task)

    # Run eval
    tasks = [task.name for task in picked] + list(original_tasks)

    tasks = list(benchmarks)
    # Useful models: meta-llama/llama-3.2-3b-instruct, openai/gpt-oss-120b
    eval_res = run_eval(
        model="openai/gpt-oss-120b",
        tasks=tasks,
        limit=10,
        num_concurrent=4,
        silent=False,
        log_debug_prompt_file=False,
        unsafe_mode=True
    )

    # Save eval result to disk
    eval_filename = f"result_eval_{make_date_string()}.json"

    eval_path = os.path.join(get_eval_dir(), eval_filename)
    with open(eval_path, "w") as f:
        json.dump(eval_res, f, indent=4)

    logger.info(f"Wrote eval results from evaluating {tasks} to file {eval_path}")
