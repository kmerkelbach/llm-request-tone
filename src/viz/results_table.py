from typing import List, Dict, Optional
import os
import json
from glob import glob

import pandas as pd

from ..evaluation.dto import EvalResult
from ..util.constants import *


class TableMaker:
    def __init__(self, results_dir: str) -> None:
        results_files = sorted(glob(os.path.join(results_dir, "*.json")))
        results_file = results_files[-1]  # most recent file

        # Parse results as EvalResult
        result_set = self._parse_results_file(results_file)

        # Make results table
        # - model
        # - benchmark
        # - scenario
        result_df = self._make_results_table(results=result_set)

    @staticmethod
    def _parse_results_file(eval_path: str) -> List[EvalResult]:
        # Open file
        with open(eval_path, "r") as f:
            loaded_eval = json.load(f)

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

    @staticmethod
    def _pick_metric(res: Dict) -> str:
        metrics = set(res.keys())

        for m in [
            "exact_match,flexible-extract",
            "pass_at_1,extract_code"
        ]:
            if m in metrics:
                return m

    def _make_results_table(self, results: List[EvalResult]) -> pd.DataFrame:
        # Build DataFrame by first constructing all the rows
        rows = []

        for res in results:
            for benchmark_name_templated, results_dict in res.results['results'].items():
                benchmark_variation = benchmark_name_templated.replace(res.benchmark_base, "")

                if benchmark_variation == "":
                    benchmark_variation = TASK_BASE

                metric = self._pick_metric(results_dict)
                val = results_dict[metric]

                rows.append({
                    FIELD_MODEL: res.model,
                    FIELD_BENCHMARK: res.benchmark_base,
                    FIELD_SCENARIO: benchmark_variation,
                    FIELD_METRIC_NAME: metric,
                    FIELD_METRIC_VALUE: val
                })

        # Make "flat" DataFrame
        df = pd.DataFrame(rows)
        return df
