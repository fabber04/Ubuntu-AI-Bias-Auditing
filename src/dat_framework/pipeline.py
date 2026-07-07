"""
End-to-end DAT framework pipeline: glues data collection, fairness diagnostics,
statistical validation, and the Tier 1-3 MOO framework into one callable.

    from dat_framework.pipeline import run_full_pipeline
    results = run_full_pipeline()

See app/streamlit_app.py for the interactive front-end built on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dat_framework.config import FairnessWeights, PROTECTED_ATTRIBUTES
from dat_framework.data.synthetic_dhs import SyntheticDataConfig, generate_dataset
from dat_framework.metrics.fairness_metrics import (
    intersectional_metrics,
    single_attribute_metrics,
)
from dat_framework.metrics.stats_tests import factorial_anova, non_additivity_check, paired_ttests
from dat_framework.optimization.moo_engine import run_w0_sweep
from dat_framework.optimization.pareto import extract_pareto_front, solutions_to_dataframe
from dat_framework.optimization.preprocessing import assign_intersectional_subgroups


@dataclass
class PipelineResults:
    data: pd.DataFrame
    baseline_single: pd.DataFrame
    baseline_intersectional: pd.DataFrame
    corrected_single: pd.DataFrame
    corrected_intersectional: pd.DataFrame
    ttest_results: pd.DataFrame
    anova_table: pd.DataFrame
    pareto_df: pd.DataFrame


def run_full_pipeline(
    n_records: int = 2000,
    seed: int = 42,
    weights: Optional[FairnessWeights] = None,
    min_group_size: int = 5,
    run_moo: bool = True,
    progress_callback=None,
) -> PipelineResults:
    weights = weights or FairnessWeights()

    # --- Phase I & data collection ---------------------------------------
    df = generate_dataset(SyntheticDataConfig(n_records=n_records, seed=seed))
    df = assign_intersectional_subgroups(df, PROTECTED_ATTRIBUTES)

    # --- Phase II: fairness diagnostics, baseline vs single-axis fix -----
    baseline_single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    baseline_inter = intersectional_metrics(
        df, y_pred_col="y_pred_baseline", min_group_size=min_group_size
    )
    corrected_single = single_attribute_metrics(df, y_pred_col="y_pred_single_axis")
    corrected_inter = intersectional_metrics(
        df, y_pred_col="y_pred_single_axis", min_group_size=min_group_size
    )

    # --- Statistical validation -------------------------------------------
    ttest_results = paired_ttests(baseline_single, baseline_inter)
    anova_table = factorial_anova(df, outcome_col="y_pred_baseline")

    # --- Tier 1-3: MOO framework -------------------------------------------
    pareto_df = pd.DataFrame()
    if run_moo:
        solutions = run_w0_sweep(
            df, weights, min_group_size=min_group_size, progress_callback=progress_callback
        )
        solutions_df = solutions_to_dataframe(solutions)
        pareto_df = extract_pareto_front(solutions_df)

    return PipelineResults(
        data=df,
        baseline_single=baseline_single,
        baseline_intersectional=baseline_inter,
        corrected_single=corrected_single,
        corrected_intersectional=corrected_inter,
        ttest_results=ttest_results,
        anova_table=anova_table,
        pareto_df=pareto_df,
    )


if __name__ == "__main__":
    results = run_full_pipeline(run_moo=True)
    print("=== Baseline single-attribute metrics ===")
    print(results.baseline_single)
    print("\n=== Baseline intersectional metrics (first 10) ===")
    print(results.baseline_intersectional.head(10))
    print("\n=== t-test: single vs intersectional ===")
    print(results.ttest_results)
    print("\n=== Factorial ANOVA ===")
    print(results.anova_table)
    print("\n=== Pareto front ===")
    print(results.pareto_df)
