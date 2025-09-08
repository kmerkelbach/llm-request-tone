from typing import List, Dict, Optional
import os
import json
from glob import glob

from ..evaluation.dto import EvalResult


class TableMaker:
    def __init__(self, results_dir: str) -> None:
        results_files = sorted(glob(os.path.join(results_dir, "*.json")))
        results_file = results_files[-1]  # most recent file

        # Open file
        with open(results_file, "r") as f:
            res = json.load(f)

        # Parse results as EvalResult
        result_set = self._make_results(res)

    @staticmethod
    def _make_results(loaded_eval: Dict) -> List[EvalResult]:
        res = []

        for model_name, model_res in loaded_eval.items():
            for benchmark_name, benchmark_res in model_res.items():

                # Skip if not valid
                result = benchmark_res['res']
                if result is None:
                    continue

                res.append(
                    EvalResult(
                        model=model_name,
                        benchmark_base=benchmark_name,
                        results=result
                    )
                )

        return res
