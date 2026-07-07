"""
Tier 3: Ubuntu-Based Pareto Frontier Selection (The Governance Layer).

Extracts the non-dominated set from the w0 sweep and exposes helpers for
human-in-the-loop selection — the paper is explicit that "final model
selection cannot be designated to automated computation" (section 3.3/Tier 3):
a human, guided by Ubuntu/Botho, weighs equity against utility loss.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from dat_framework.optimization.moo_engine import MOOSolution


def solutions_to_dataframe(solutions: List[MOOSolution]) -> pd.DataFrame:
    rows = [
        {
            "w0": s.w0,
            "mean_DIR": s.mean_dir,
            "mean_DPD": s.mean_dpd,
            "mean_EOD": s.mean_eod,
            "fairness_loss": s.fairness_loss,
            "accuracy_pct": 100 * s.accuracy,
            "utility_retention_pct": s.utility_retention_pct,
        }
        for s in solutions
    ]
    return pd.DataFrame(rows)


def extract_pareto_front(solutions_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the non-dominated solutions: no other solution has both
    lower (or equal) fairness_loss AND higher (or equal) accuracy, with at
    least one strictly better — the formal Pareto-optimal set P defined in
    the paper's Tier 3 section.
    """
    df = solutions_df.dropna(subset=["fairness_loss", "accuracy_pct"]).copy()
    is_pareto = np.ones(len(df), dtype=bool)
    acc = df["accuracy_pct"].to_numpy()
    loss = df["fairness_loss"].to_numpy()

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            dominates = (
                acc[j] >= acc[i] and loss[j] <= loss[i]
                and (acc[j] > acc[i] or loss[j] < loss[i])
            )
            if dominates:
                is_pareto[i] = False
                break

    df["is_pareto_optimal"] = is_pareto
    return df.sort_values("w0").reset_index(drop=True)


def recommend_solution(
    pareto_df: pd.DataFrame,
    utility_floor_pct: float = 95.0,
    priority: str = "balanced",
) -> pd.Series:
    """A *suggestion*, not a decision — the paper is emphatic that final
    selection stays with human stakeholders (Ubuntu/Botho governance). This
    just pre-filters to help a person start their deliberation.

    priority: "balanced" (lowest fairness_loss subject to utility floor),
              "utility" (highest accuracy among Pareto-optimal solutions),
              "fairness" (lowest fairness_loss regardless of utility floor).
    """
    candidates = pareto_df[pareto_df["is_pareto_optimal"]]
    if candidates.empty:
        candidates = pareto_df

    if priority == "utility":
        return candidates.loc[candidates["accuracy_pct"].idxmax()]
    if priority == "fairness":
        return candidates.loc[candidates["fairness_loss"].idxmin()]

    # balanced (default): among solutions meeting the utility floor, pick
    # lowest fairness loss; if none meet the floor, take the least-bad
    # trade-off (closest to the floor).
    meets_floor = candidates[candidates["utility_retention_pct"] >= utility_floor_pct]
    if not meets_floor.empty:
        return meets_floor.loc[meets_floor["fairness_loss"].idxmin()]
    diff = (candidates["utility_retention_pct"] - utility_floor_pct).abs()
    return candidates.loc[diff.idxmin()]
