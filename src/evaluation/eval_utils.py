from typing import List, Dict, Optional
import os
import json
from glob import glob

from ..evaluation.dto import EvalResult
from ..util.utils import get_eval_dir
from ..util.constants import *


def make_eval_key(model: str, benchmark: str) -> str:
    return model + "_" + benchmark


def load_result_file(eval_path: str, res: Optional[Dict[str, EvalResult]]) -> Dict[str, EvalResult]:
    # Open file
    with open(eval_path, "r") as f:
        loaded_eval = json.load(f)

    # If no results supplied, start fresh
    if not res:
        res: Dict[str, EvalResult] = {}

    for model_name, model_res in loaded_eval.items():
        for benchmark_name, benchmark_res in model_res.items():

            # Skip if not valid
            result = benchmark_res['res']
            if result is None:
                continue

            key = make_eval_key(
                model=model_name,
                benchmark=benchmark_name
            )

            res[key] = EvalResult(
                model=model_name,
                benchmark_base=benchmark_name,
                results=result
            )

    return res


def load_results_from_dir(dir_path: str) -> Dict[str, EvalResult]:
    results_files = sorted(glob(os.path.join(dir_path, "*.json")))

    # Load results from old to new, overwriting combos of model and benchmark if newer data exists
    res = {}
    for file_path in results_files:
        res = load_result_file(file_path, res=res)

    return res