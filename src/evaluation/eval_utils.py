from collections import defaultdict
from typing import List, Dict, Optional
import os
import json
from glob import glob

from ..evaluation.dto import EvalResult


def make_eval_key(model: str, benchmark: str) -> str:
    return model + "_" + benchmark


def load_result_file(eval_path: str, res: Optional[Dict[str, List[EvalResult]]]) -> Dict[str, List[EvalResult]]:
    # Open file
    with open(eval_path, "r") as f:
        loaded_eval = json.load(f)

    # If no results supplied, start fresh
    if not res:
        res: Dict[str, List[EvalResult]] = defaultdict(list)

    # Load results
    for combo_key, loaded in loaded_eval.items():

        # Interpret dict as dataclass
        res[combo_key].append(EvalResult.from_dict(loaded))

    return res


def load_results_from_dir(dir_path: str) -> Dict[str, List[EvalResult]]:
    results_files = sorted(glob(os.path.join(dir_path, "*.json")))

    # Load results
    res = {}
    for file_path in results_files:
        res = load_result_file(file_path, res=res)

    return res