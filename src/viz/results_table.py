from typing import List, Dict, Optional
import os
import json
from glob import glob
from itertools import product

import numpy as np
import pandas as pd

from ..evaluation.dto import EvalResult
from ..evaluation.eval_utils import load_results_from_dir
from ..util.utils import get_tables_dir, mkdir
from ..util.constants import *


class TableMaker:
    def __init__(self, results_dir: str) -> None:
        # Parse results as EvalResult
        result_dict = load_results_from_dir(results_dir)

        # Make results table
        # - model
        # - benchmark
        # - scenario
        self.results_df = self._make_results_table(results=result_dict)

        # Make different aggregations of the table
        self._aggregate_table()

    def _aggregate_table(self):
        for framework in [FRAMEWORK_LM_EVAL, FRAMEWORK_SORRY]:
            framework_dir = mkdir(
                os.path.join(get_tables_dir(), framework)
            )

            df_scenario_vs_model = self._aggregate_by_scenario_and_model(
                self.results_df.copy(),
                framework_filter=framework
            )
            df_scenario_vs_model.to_csv(
                os.path.join(framework_dir, "scenario_vs_model.csv")
            )

            df_scenario_vs_benchmark = self._aggregate_by_scenario_and_benchmark(
                self.results_df.copy(),
                framework_filter=framework
            )
            df_scenario_vs_benchmark.to_csv(
                os.path.join(framework_dir, "scenario_vs_benchmark.csv")
            )

    def _subtract_baseline(self, df: pd.DataFrame, framework_filter: str) -> pd.DataFrame:
        # Let's subtract the base task performance for each model/benchmark combination
        normalized_to_base = []
        models = np.unique(df[FIELD_MODEL])
        benchmarks = np.unique(df[FIELD_BENCHMARK])
        for model, benchmark in product(models, benchmarks):
            # Find out base scenario's value
            selection = df[(df[FIELD_MODEL] == model)
                           & (df[FIELD_BENCHMARK] == benchmark)]
            base_selection = selection[(selection[FIELD_SCENARIO] == TASK_BASELINE)]
            base_val = base_selection.iloc[0][FIELD_METRIC_VALUE]

            # Skip if framework is incorrect
            framework = base_selection.iloc[0][FIELD_FRAMEWORK]
            if framework != framework_filter:
                continue

            # Subtract it everywhere (for this combo)
            selection[FIELD_METRIC_VALUE] -= base_val
            normalized_to_base.append(selection)
        normalized_to_base = pd.concat(normalized_to_base)

        return normalized_to_base

    def _aggregate_by_scenario_and_benchmark(self, df: pd.DataFrame, framework_filter: str) -> pd.DataFrame:
        normalized_to_base = self._subtract_baseline(df, framework_filter=framework_filter)

        # Make pivot table
        df_res = pd.pivot_table(
            normalized_to_base,
            values=[FIELD_METRIC_VALUE],
            index=[FIELD_SCENARIO],
            columns=[FIELD_BENCHMARK],
            aggfunc="mean",
            margins=False
        )

        return df_res

    def _aggregate_by_scenario_and_model(self, df: pd.DataFrame, framework_filter: str) -> pd.DataFrame:
        normalized_to_base = self._subtract_baseline(df, framework_filter=framework_filter)

        # Make pivot table
        df_res = pd.pivot_table(
            normalized_to_base,
            values=[FIELD_METRIC_VALUE],
            index=[FIELD_SCENARIO],
            columns=[FIELD_MODEL],
            aggfunc="mean",
            margins=True
        )
        df_res = df_res.iloc[:-1]  # remove lower margin

        return df_res

    @staticmethod
    def _pick_metric(res: Dict) -> str:
        metrics = set(res.keys())

        for m in [
            "exact_match,flexible-extract",
            "pass_at_1,extract_code",
            "bleurt_acc,none",
            "bleu_acc,none"
        ]:
            if m in metrics:
                return m

    def _make_results_table(self, results: Dict[str, EvalResult]) -> pd.DataFrame:
        # Build DataFrame by first constructing all the rows
        rows = []

        for res in results.values():
            for benchmark_name_templated, results_dict in res.results.items():
                benchmark_variation = benchmark_name_templated.replace(res.benchmark_base, "")

                if benchmark_variation.startswith(TEMPLATED_STR):
                    benchmark_variation = benchmark_variation[len(TEMPLATED_STR):]
                benchmark_variation = benchmark_variation.strip("_")

                if res.framework == FRAMEWORK_SORRY:
                    # Report average compliance
                    compliance_rates = list(results_dict.values())
                    compliance_rate_avg = np.mean(compliance_rates)  # naive mean is unbaised since class counts are
                    # balanced for SORRY-Bench
                    metric = "compliance_rate"
                    val = compliance_rate_avg
                else:
                    # lm-eval
                    metric = self._pick_metric(results_dict)
                    val = float(results_dict[metric])

                rows.append({
                    FIELD_MODEL: res.model,
                    FIELD_BENCHMARK: res.benchmark_base,
                    FIELD_FRAMEWORK: res.framework,
                    FIELD_SCENARIO: benchmark_variation,
                    FIELD_METRIC_NAME: metric,
                    FIELD_METRIC_VALUE: val
                })

        # Make "flat" DataFrame
        df = pd.DataFrame(rows)
        return df
