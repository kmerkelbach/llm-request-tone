import json
import os
import random

from loguru import logger
from typing import List, Dict, Optional

from .evaluation.lm_eval_shell import run_lm_eval
from .evaluation.sorry_bench_shell import run_sorry_bench
from .framing.task_framer import TaskFramer
from .framing.dto import ModifiedTask
from .util.utils import get_eval_dir, make_date_string
from .util.constants import *


benchmarks_all = [
    "mmlu_pro",
    "gpqa_diamond_cot_zeroshot",
    "gsm8k_cot_llama",
    "ifeval",
    "truthfulqa_gen",
    "mbpp_plus_instruct",
    "bbh_cot_zeroshot",
    SORRY_BENCH_NAME
]


def write_results(eval_res: Dict) -> str:
    eval_filename = f"result_eval_{make_date_string()}.json"

    eval_path = os.path.join(get_eval_dir(), eval_filename)
    with open(eval_path, "w") as f:
        json.dump(eval_res, f, indent=4)

    logger.info(f"Wrote eval results to file {eval_path}")

    return eval_filename


def run_eval_for_benchmark_and_framings(framed_tasks: List[ModifiedTask], base_benchmark: str,
                                        model: str = "openai/gpt-oss-20b", limit: Optional[int] = 10,
                                        num_concurrent: Optional[int] = 16,
                                        write_to_disk: Optional[bool] = True,
                                        silent: bool = False) -> Dict:
    # Get task set
    filtered = [task for task in framed_tasks if base_benchmark in task.name]

    # Make sure all the tasks we run are templated - even the unchanged "baseline" one should be
    # templated to ensure comparability.
    assert all(TEMPLATED_STR in task.name for task in filtered), f"All tasks must be templated! -> {filtered}"

    # Separate the tasks into SORRY-Bench tasks and lm-eval tasks
    tasks_lm_eval = [task for task in filtered if task.origin_task != SORRY_BENCH_NAME]
    tasks_sorry = [task for task in filtered if task.origin_task == SORRY_BENCH_NAME]

    eval_res = {
        FRAMEWORK_LM_EVAL: None,
        FRAMEWORK_SORRY: None
    }

    if len(tasks_lm_eval) > 0:
        eval_res[FRAMEWORK_LM_EVAL] = run_lm_eval(
            model=model,
            tasks=[task.name for task in tasks_lm_eval],
            limit=limit,
            num_concurrent=num_concurrent,
            silent=silent,
            log_debug_prompt_file=True,
            unsafe_mode=True
        )

    if len(tasks_sorry) > 0:
        eval_res[FRAMEWORK_SORRY] =run_sorry_bench(
            model=model,
            data_mutations=[task.scenario.name for task in tasks_sorry],
            parallel=num_concurrent,
            silent=silent
        )

    # Save eval result to disk
    if write_to_disk:
        write_results(eval_res)

    return eval_res


if __name__ == "__main__":
    # Frame tasks
    framer = TaskFramer()
    modified_tasks: List[ModifiedTask] = framer.template_all_tasks()

    # Define benchmarks
    benchmarks_subset = [
        # "mmlu_pro",
        # "gpqa_diamond_cot_zeroshot",
        # "gsm8k_cot_llama",
        # "ifeval",
        # "truthfulqa_gen",
        # "mbpp_plus_instruct",
        # "bbh_cot_zeroshot",
        SORRY_BENCH_NAME
    ]

    # Make sure the selected base benchmarks are all in the main benchmark list
    assert all(b in benchmarks_all for b in benchmarks_subset), "Not all selected benchmarks are in the main list!"

    # Run eval for different models
    models = [
        # "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        # "meta-llama/llama-3.2-3b-instruct",
        # "qwen/qwen3-30b-a3b-thinking-2507",
        # "x-ai/grok-code-fast-1",
        # "anthropic/claude-sonnet-4",
        # "deepseek/deepseek-chat-v3-0324"
    ]
    results: Dict[str, Dict[str, Dict]] = {}
    # results[model_name][benchmark_name] = ... (eval res)

    for model in models:

        model_res = {}
        for bench in benchmarks_subset:
            try:
                res = run_eval_for_benchmark_and_framings(
                    framed_tasks=modified_tasks,
                    base_benchmark=bench,
                    model=model,
                    num_concurrent=32,
                    limit=2,
                    write_to_disk=False,
                )
                msg = "OK"
            except RuntimeError as e:
                res = None
                msg = str(e)

            model_res[bench] = {
                "res": res,
                "msg": msg
            }

        results[model] = model_res

    # Write overall results to disk
    write_results(results)
