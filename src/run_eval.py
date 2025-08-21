import json
import os
import random

from loguru import logger
from typing import List

from .evaluation.lm_eval_shell import run_eval
from .framing.task_framer import TaskFramer
from .framing.dto import ModifiedTask
from .util.utils import get_eval_dir, make_date_string


if __name__ == "__main__":
    # Frame tasks
    framer = TaskFramer()
    modified_tasks: List[ModifiedTask] = framer.template_all_tasks()

    # For testing, pick a templated tasks at random
    original = random.choice(["gsm8k", "mmlu_pro"])
    picked: ModifiedTask = random.choice([task for task in modified_tasks if task.origin_task == original])

    # Run eval
    tasks = [original, picked.name]
    eval_res = run_eval(
        model="openai/gpt-oss-120b",
        tasks=tasks,
        limit=1,
        num_concurrent=16,
        silent=False,
        log_debug_prompt_file=True
    )

    # Save eval result to disk
    eval_filename = f"result_eval_{make_date_string()}.json"

    eval_path = os.path.join(get_eval_dir(), eval_filename)
    with open(eval_path, "w") as f:
        json.dump(eval_res, f, indent=4)

    logger.info(f"Wrote eval results from evaluating {tasks} to file {eval_path}")
