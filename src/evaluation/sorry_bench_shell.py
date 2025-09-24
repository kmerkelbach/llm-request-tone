import json
import os
from glob import glob
import random
import string
import subprocess
from typing import Optional, Union, Sequence, Dict, List

from interegular.fsm import parallel
from loguru import logger
from textwrap import wrap
from pydoc import importfile

from ..util.constants import *
from ..util.utils import read_jsonl


def run_sorry_bench(model: str, data_mutations: List[str], parallel: int = 4, bench_name: str = SORRY_TEMPLATED_NAME,
                    judge_model: str = "openai/gpt-4", silent=True) -> Dict:
    # Find SORRY-Bench dir
    sorry_dir = os.path.split(os.path.split(os.path.split(PATH_SORRY_BENCH_QUESTIONS)[0])[0])[0]
    orig_dir = os.getcwd()
    os.chdir(sorry_dir)

    results = {}
    for mutation in data_mutations:
        # Run model on SORRY-Bench prompts
        cmd = [
            "python",
            "gen_api_answer.py",
            "--model",
            model,
            "--parallel",
            str(parallel),
            "--bench-name",
            bench_name,
            "--data-mutation",
            mutation,
        ]
        run_script(cmd, silent=silent)

        # Run judge on model answers
        # python gen_judgment_safety.py  --bench-name sorry_bench --judge-model "openai/gpt-4" --model-list "openai/gpt-oss-20b" --parallel 16
        cmd = [
            "python",
            "gen_judgment_safety.py",
            "--bench-name",
            bench_name,
            "--judge-model",
            judge_model,
            "--model-list",
            model,
            "--parallel",
            str(parallel),
            "--data-mutation",
            mutation,
        ]
        run_script(cmd, silent=silent)

        # Load output file produced by judge
        judge_output_filter = os.path.join(
            "data",
            SORRY_TEMPLATED_NAME,
            f"model_judgment_{mutation}",
            f"{judge_model}*.jsonl"
        )
        judge_output_path = glob(judge_output_filter)[0]
        judgments: List[Dict] = read_jsonl(judge_output_path)

        # Interpret refusals
        # In SORRY-Bench judging, score of 1 means the AI complied with the harmful request and a score of 0 means a refusal
        judgments = [{"question_id": jud["question_id"], FIELD_REFUSED: True if jud["score"] == 0 else False} for jud in judgments]

        # Find out question category
        categories = load_category_file(sorry_dir)
        for jud in judgments:
            category_index = (jud["question_id"] - 1) // 10
            jud[FIELD_CATEGORY] = categories[category_index]
            jud.pop("question_id")

        results[mutation] = judgments

    # Go back to original dir
    os.chdir(orig_dir)

    return results


def load_category_file(sorry_dir: str) -> List[str]:
    meta_info_path = os.path.join(sorry_dir, "data", SORRY_BENCH_NAME, "meta_info.py")
    meta_info = importfile(meta_info_path)
    return meta_info.category_descriptions_shortest


def run_script(command: List[str], silent: bool) -> subprocess.CompletedProcess:
    env = os.environ.copy()

    result = subprocess.run(
        command,
        env=env,
        check=False,
        capture_output=silent,
    )

    return result
