import json
import os
import itertools
from datetime import datetime

from loguru import logger
from typing import List, Dict, Optional

from .evaluation.lm_eval_shell import run_lm_eval
from .evaluation.sorry_bench_shell import run_sorry_bench
from .evaluation.eval_utils import load_results_from_dir, make_eval_key
from .evaluation.dto import EvalResult
from .framing.task_framer import TaskFramer
from .framing.dto import ModifiedTask
from .util.utils import get_eval_dir, make_date_string
from src.evaluation.config import benchmarks_selected, models, lm_eval_limit, temperature
from .util.constants import *


def write_results(eval_results: Dict[str, EvalResult]) -> str:
    # Convert from EvalResult to dict
    eval_results = {k: v.to_dict() for (k, v) in eval_results.items()}

    eval_filename = f"result_eval_{make_date_string()}.json"

    eval_path = os.path.join(get_eval_dir(), eval_filename)
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    logger.info(f"Wrote eval results to file {eval_path}")

    return eval_filename


def run_eval_for_benchmark_and_framings(framed_tasks: List[ModifiedTask], base_benchmark: str,
                                        model: str = "openai/gpt-oss-20b", limit: Optional[int] = 10,
                                        num_concurrent: Optional[int] = 16,
                                        write_to_disk: Optional[bool] = True,
                                        silent: bool = False) -> None:
    # Get task set
    filtered = [task for task in framed_tasks if base_benchmark in task.name]

    # Make sure all the tasks we run are templated - even the unchanged "baseline" one should be
    # templated to ensure comparability.
    assert all(TEMPLATED_STR in task.name for task in filtered), f"All tasks must be templated! -> {filtered}"

    # Separate the tasks into SORRY-Bench tasks and lm-eval tasks
    tasks_lm_eval = [task for task in filtered if task.origin_task != SORRY_BENCH_NAME]
    tasks_sorry = [task for task in filtered if task.origin_task == SORRY_BENCH_NAME]

    # Make sure we are only running either SORRY or lm-eval
    assert (len(tasks_lm_eval) > 0) ^ (len(tasks_sorry) > 0), "Can only run SORRY-Bench or lm-eval, NOT BOTH."

    res: Optional[Dict] = None
    framework: Optional[str] = None

    if len(tasks_lm_eval) > 0:
        framework = FRAMEWORK_LM_EVAL
        res = run_lm_eval(
            model=model,
            tasks=[task.name for task in tasks_lm_eval],
            limit=limit,
            num_concurrent=num_concurrent,
            silent=silent,
            log_debug_prompt_file=False,
            unsafe_mode=True,
            temperature=temperature
        )

    if len(tasks_sorry) > 0:
        framework = FRAMEWORK_SORRY
        res = run_sorry_bench(
            model=model,
            data_mutations=[task.scenario.name for task in tasks_sorry],
            parallel=num_concurrent,
            silent=silent
        )

    eval_res = EvalResult(
        model=model,
        benchmark_base=base_benchmark,
        framework=framework,
        results=res,
        date_created=datetime.isoformat(datetime.now())
    )

    # Save eval result to disk
    if write_to_disk:
        eval_dict = {
            make_eval_key(model=model, benchmark=base_benchmark): eval_res
        }
        write_results(eval_dict)


def run_eval(force_run: bool = False):
    # Frame tasks
    framer = TaskFramer()
    modified_tasks: List[ModifiedTask] = framer.template_all_tasks()

    # Make all benchmark/model combos
    combos = list(itertools.product(models, benchmarks_selected))
    num_combos = len(combos)
    logger.info(f"Found {num_combos} combinations of model and benchmark.")

    # Load existing results
    results_loaded: Dict[str, EvalResult] = load_results_from_dir(get_eval_dir())

    for idx, (model, benchmark) in enumerate(combos):
        logger.info(f"Combination {idx + 1} of {num_combos}: MODEL {model}; BENCHMARK {benchmark}")

        # Skip if we already have data on this combination
        combo_key = make_eval_key(
            model=model,
            benchmark=benchmark
        )
        if combo_key in results_loaded and not force_run:
            logger.info(f"Skipping (we already have data)")
            continue

        # Run the benchmark with the model
        try:
            run_eval_for_benchmark_and_framings(
                framed_tasks=modified_tasks,
                base_benchmark=benchmark,
                model=model,
                num_concurrent=32,
                limit=lm_eval_limit,
                write_to_disk=True  # critical: otherwise, nothing is saved
            )
        except RuntimeError as e:
            logger.warning(f"Could not run combo {model} / {benchmark}.")


if __name__ == "__main__":
    run_eval()
