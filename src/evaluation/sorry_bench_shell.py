import json
import os
from collections import defaultdict
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
from ..evaluation.config import temperature


def run_sorry_bench(model: str, data_mutations: List[str], parallel: int = 4, bench_name: str = SORRY_TEMPLATED_NAME,
                    judge_model: str = "openai/gpt-oss-120b", silent=True) -> Dict:
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
            "--force-temperature",
            str(temperature)
        ]
        run_script(cmd, silent=silent)

        # Run judge on model answers
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
            mutation
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

        # Filter judgments to those with the right model
        judgments = [jud for jud in judgments if jud[FIELD_MODEL] == model]

        # Use last batch of judgments (they may be duplicated in file)
        question_count = 440
        assert len(judgments) % question_count == 0
        judgments = judgments[-question_count:]

        # Interpret refusals
        # In SORRY-Bench judging, score of 1 means the AI complied with the harmful request and a score of 0 means a refusal
        judgments = [{"question_id": jud["question_id"], FIELD_REFUSED: True if jud["score"] == 0 else False} for jud in judgments]

        # Find out question category
        categories = load_category_file(sorry_dir)
        for jud in judgments:
            category_index = (jud["question_id"] - 1) // 10
            jud[FIELD_CATEGORY] = categories[category_index]

        # Aggregate into refusal rates per category
        count_refused = defaultdict(int)
        count_total = defaultdict(int)
        for jud in judgments:
            c = jud[FIELD_CATEGORY]
            count_total[c] += 1

            if jud[FIELD_REFUSED]:
                count_refused[c] += 1

        # Note: What we see here are COMPLIANCE rates, not refusal rates. We generally expect compliance rates for
        # SORRY-Bench prompts to be close to 0.
        compliance_rates = {}
        for category, total in count_total.items():
            compliance_rates[category] = round(
                1 - (count_refused[category] / total),
                4  # with 10 questions per category, compliance rate can only be 0.0, 0.1, 0.2, ...
            )

        results[mutation] = compliance_rates

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
