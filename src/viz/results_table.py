from typing import List, Dict, Optional, Tuple
import os
import json
from glob import glob
from itertools import product
from loguru import logger

import numpy as np
import pandas as pd

from ..evaluation.dto import EvalResult
from ..framing.task_framer import TaskFramer
from ..evaluation.eval_utils import load_results_from_dir
from ..util.utils import get_tables_dir, mkdir
from ..util.constants import *
from ..evaluation.config import models, benchmarks_display_names


class TableMaker:
    def __init__(self, results_dir: str) -> None:
        # Load TaskFramer - we need it for display-friendly scenario names
        self._task_framer = TaskFramer()

        # Parse results as EvalResult
        result_dict = load_results_from_dir(results_dir)

        # Make results table
        # - model
        # - benchmark
        # - scenario
        self.results_df = self._make_results_table(results=result_dict)
        self.results_df.to_csv(
            os.path.join(get_tables_dir(), "results_full.csv")
        )

        if len(self.results_df) == 0:
            logger.info("No results loaded. Exiting.")
            return

        # Make different aggregations of the table
        self._aggregate_table()

    def _aggregate_table(self):
        # Columns to aggregate - each entry is a list of columns that we will use for aggregation (i.e., aggregating
        # over the _other_ columns).
        column_sets = [
            [FIELD_MODEL],
            [FIELD_BENCHMARK],
            [FIELD_MODEL_FAMILY],
            [FIELD_MODEL_SIZE],
            [FIELD_BENCHMARK, FIELD_MODEL],
            [FIELD_BENCHMARK, FIELD_MODEL_FAMILY],
        ]

        for framework in [FRAMEWORK_LM_EVAL, FRAMEWORK_SORRY]:
            framework_dir = mkdir(
                os.path.join(get_tables_dir(), framework)
            )

            for col_set in column_sets:
                df_agg = self._aggregate_by_scenario(
                    self.results_df.copy(),
                    framework_filter=framework,
                    columns_to_show=col_set
                )
                col_set_str = "_".join(col_set)
                filename_base = f"scenario_vs_{col_set_str}"

                # Save with and without "All" columns
                for with_all in [True, False]:
                    if not with_all:
                        df_agg = df_agg[df_agg.columns[:-1]]

                    var_dir = mkdir(
                        os.path.join(framework_dir, "with_all" if with_all else "plain")
                    )

                    df_agg.to_csv(os.path.join(var_dir, filename_base + ".csv"), index=True)
                    df_agg.to_markdown(os.path.join(var_dir, filename_base + ".md"), index=True)

    def _divide_by_baseline(self, df: pd.DataFrame, framework_filter: str) -> pd.DataFrame:
        # Let's divide by the base task performance for each model/benchmark combination
        normalized_to_base = []
        models = np.unique(df[FIELD_MODEL])
        benchmarks = np.unique(df[FIELD_BENCHMARK])
        for model, benchmark in product(models, benchmarks):
            # Find out base scenario's value
            selection = df[(df[FIELD_MODEL] == model)
                           & (df[FIELD_BENCHMARK] == benchmark)]
            selection = selection.reset_index(drop=True)  # would otherwise get a warning when changing value
            base_selection = selection[(selection[FIELD_SCENARIO] == TASK_BASELINE_DISPLAY)]
            base_val = base_selection.iloc[0][FIELD_METRIC_VALUE]

            # Skip if framework is incorrect
            framework = base_selection.iloc[0][FIELD_FRAMEWORK]
            if framework != framework_filter:
                continue

            # Divide by it everywhere (for this combo)
            selection[FIELD_METRIC_VALUE] /= base_val
            normalized_to_base.append(selection)
        normalized_to_base = pd.concat(normalized_to_base)

        return normalized_to_base

    def _aggregate_by_scenario(self, df: pd.DataFrame, framework_filter: str,
                               columns_to_show: List[str],
                               show_change_as_percentage: bool = True) -> pd.DataFrame:
        normalized_to_base = self._divide_by_baseline(df, framework_filter=framework_filter)

        # Optionally convert percentages to changes (e.g., 0.9 to -10% and 1.3 to +30%)
        if show_change_as_percentage:
            normalized_to_base[FIELD_METRIC_VALUE] = normalized_to_base[FIELD_METRIC_VALUE].map(
                lambda val: 100 * (val - 1)  # e.g., 0.9 -> -10
            )

        # Capitalize scenario column
        scenario_col = FIELD_SCENARIO.capitalize()
        normalized_to_base = normalized_to_base.rename(
            columns={FIELD_SCENARIO: scenario_col}
        )

        # Apply nice display names for benchmarks
        normalized_to_base[FIELD_BENCHMARK] = normalized_to_base[FIELD_BENCHMARK].apply(
            lambda bench_name: benchmarks_display_names[bench_name]
        )

        # Make pivot table
        df_res = pd.pivot_table(
            normalized_to_base,
            values=[FIELD_METRIC_VALUE],
            index=[scenario_col],
            columns=columns_to_show,
            aggfunc="median",
            margins=True
        )
        df_res = df_res.iloc[:-1]  # remove lower margin

        # Remove superfluous "value" part of columns
        df_res.columns = df_res.columns.droplevel()

        def flatten_entries(entries: Tuple) -> str:
            n = len(entries)
            if n == 2:
                return f"{entries[0]} with {entries[1]}"
            elif n == 3:
                return f"{entries[0]} with {entries[1]} and {entries[2]}"
            else:
                head = [str(e) for e in entries[:-1]]
                tail = str(entries[-1])
                return ", ".join(head) + ", and " + tail

        # Flatten MultiIndex
        if len(columns_to_show) > 1:
            idx = df_res.columns.to_series()
            idx = idx.apply(flatten_entries)
            df_res.columns = idx

        # Format changes
        if show_change_as_percentage:
            df_res = df_res.iloc[1:]  # remove baseline row
            df_res = df_res.map(lambda val: f"{'+' if val >= 0 else ''}{val:0.1f}%")  # e.g., 13 -> "+13%"

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

    @staticmethod
    def _get_model_family(model_name: str) -> str:
        model_name = model_name.lower()
        if "gpt" in model_name:
            return "GPT"
        elif "qwen" in model_name:
            return "Qwen"
        elif "llama" in model_name:
            return "Llama"
        else:
            assert False, f"Cannot determine model family for {model_name}"

    @staticmethod
    def _get_model_size(model_id: str) -> str:
        model_id = model_id.lower()

        # Get last part of name
        model_name = model_id.split("/")[-1]

        # Find a part of the name that ends with "b"
        size_specifiers = [tok for tok in model_name.split("-") if tok.endswith("b")]
        assert len(size_specifiers) == 1, f"Could not get model size from model ID {model_id}"

        # Extract size (in billions of parameters)
        size = int(size_specifiers[0].strip("b"))
        if size < 35:
            return MODEL_SIZE_SMALL
        else:
            return MODEL_SIZE_LARGE

    def _make_results_table(self, results: Dict[str, EvalResult]) -> pd.DataFrame:
        # Build DataFrame by first constructing all the rows
        rows = []

        for res_key, res in results.items():

            # Ignore this result if the model does not appear in our model list
            model = res.model
            if model not in models:
                logger.info(f"Skipping result with key '{res_key}': Model {model} not included in models list.")
                continue

            # Find out model family and size
            model_family = self._get_model_family(model)
            model_size = self._get_model_size(model)

            for benchmark_name_templated, results_dict in res.results.items():
                benchmark_variation = benchmark_name_templated.replace(res.benchmark_base, "")

                if benchmark_variation.startswith(TEMPLATED_STR):
                    benchmark_variation = benchmark_variation[len(TEMPLATED_STR):]
                benchmark_variation = benchmark_variation.strip("_")

                scenario_name = self._task_framer.get_display_name(scenario_name=benchmark_variation)

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
                    FIELD_MODEL: model,
                    FIELD_MODEL_FAMILY: model_family,
                    FIELD_MODEL_SIZE: model_size,
                    FIELD_BENCHMARK: res.benchmark_base,
                    FIELD_FRAMEWORK: res.framework,
                    FIELD_SCENARIO: scenario_name,
                    FIELD_METRIC_NAME: metric,
                    FIELD_METRIC_VALUE: val
                })

        # Make "flat" DataFrame
        df = pd.DataFrame(rows)

        # Sort
        df.sort_values(
            by=[FIELD_MODEL_FAMILY, FIELD_MODEL_SIZE, FIELD_FRAMEWORK, FIELD_BENCHMARK, FIELD_SCENARIO],
            ascending=[True, False, True, True, True],
            inplace=True
        )

        return df
