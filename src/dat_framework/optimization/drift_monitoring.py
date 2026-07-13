"""
Real-time (batch-cadence) monitoring: Population Stability Index (PSI) drift
detection and fallback triggers, per the Track 4 ToR section 2.2.

Designed for the governance-framework deployment mode described in the ToR:
"a governance framework or pilot design may instead define logs, thresholds,
review cycles, escalation triggers and reporting routines." This module gives
that a concrete, testable implementation rather than just a policy description:
run it on a schedule (e.g. weekly) comparing a fresh batch of decisions against
the baseline reference distribution, and it tells you whether input drift or
fairness drift has crossed an actionable threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from dat_framework.metrics.fairness_metrics import intersectional_metrics, single_attribute_metrics

# Conventional PSI interpretation thresholds (industry-standard bands used in
# credit risk / model-ops monitoring):
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25
# > PSI_MODERATE_SHIFT is considered a significant shift requiring action.


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, n_buckets: int = 10
) -> float:
    """Compute the Population Stability Index between a baseline distribution
    and a current one, for a single continuous or ordinal variable.

    PSI = sum( (current_pct - baseline_pct) * ln(current_pct / baseline_pct) )
    over quantile buckets fit on the baseline sample.

    Interpretation (industry-standard bands):
      PSI < 0.10            -> no significant shift
      0.10 <= PSI < 0.25     -> moderate shift, worth investigating
      PSI >= 0.25            -> significant shift, action required
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 3:
        # Degenerate baseline (near-constant) — fall back to a coarser bucketing
        edges = np.unique(np.linspace(baseline.min(), baseline.max(), 3))
        if len(edges) < 3:
            return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    baseline_pct = np.clip(baseline_counts / max(baseline_counts.sum(), 1), 1e-4, None)
    current_pct = np.clip(current_counts / max(current_counts.sum(), 1), 1e-4, None)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


@dataclass
class DriftReport:
    feature_psi: pd.Series               # PSI per input feature
    subgroup_selection_rate_psi: pd.Series  # PSI of selection rate per intersectional subgroup
    fairness_metric_deltas: pd.DataFrame   # baseline vs. current DIR/DPD/EOD per attribute
    fallback_triggered: bool
    fallback_reasons: List[str]


def run_drift_check(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    y_pred_col: str = "y_pred_baseline",
    y_true_col: str = "y_true",
    min_group_size: int = 5,
    psi_action_threshold: float = PSI_MODERATE_SHIFT,
    fairness_delta_threshold: float = 0.05,
) -> DriftReport:
    """Compare a fresh batch (`current_df`) against the reference batch used
    at deployment/pilot sign-off (`baseline_df`). Flags a fallback trigger
    (revert to the last known-good model / rule-based baseline, per ToR 2.2)
    if either input drift or fairness drift crosses its threshold.
    """
    feature_psi = pd.Series(
        {col: population_stability_index(baseline_df[col], current_df[col]) for col in feature_cols}
    )

    baseline_single = single_attribute_metrics(baseline_df, y_pred_col=y_pred_col, y_true_col=y_true_col)
    current_single = single_attribute_metrics(current_df, y_pred_col=y_pred_col, y_true_col=y_true_col)
    deltas = pd.DataFrame({
        "baseline_DIR": baseline_single["DIR"], "current_DIR": current_single["DIR"],
        "DIR_delta": (current_single["DIR"] - baseline_single["DIR"]),
        "baseline_DPD": baseline_single["DPD"], "current_DPD": current_single["DPD"],
        "DPD_delta": (current_single["DPD"] - baseline_single["DPD"]).abs(),
    })

    # Subgroup selection-rate PSI: treat each subgroup's predicted-positive
    # rate over time as the signal being tracked (a compact proxy for
    # intersectional fairness drift without needing per-subgroup PSI on raw
    # features, which sparse subgroups can't support reliably).
    inter_baseline = intersectional_metrics(baseline_df, y_pred_col=y_pred_col, y_true_col=y_true_col,
                                             min_group_size=min_group_size)
    inter_current = intersectional_metrics(current_df, y_pred_col=y_pred_col, y_true_col=y_true_col,
                                            min_group_size=min_group_size)
    common = inter_baseline.index.intersection(inter_current.index)
    subgroup_psi = pd.Series({
        g: abs(inter_current.loc[g, "selection_rate_marginalized"] - inter_baseline.loc[g, "selection_rate_marginalized"])
        for g in common
    })

    reasons = []
    if (feature_psi > psi_action_threshold).any():
        drifted = list(feature_psi[feature_psi > psi_action_threshold].index)
        reasons.append(f"Input drift: PSI > {psi_action_threshold} for {drifted}")
    if (deltas["DPD_delta"] > fairness_delta_threshold).any():
        worsened = list(deltas[deltas["DPD_delta"] > fairness_delta_threshold].index)
        reasons.append(f"Fairness drift: DPD moved > {fairness_delta_threshold} for {worsened}")

    return DriftReport(
        feature_psi=feature_psi,
        subgroup_selection_rate_psi=subgroup_psi,
        fairness_metric_deltas=deltas,
        fallback_triggered=len(reasons) > 0,
        fallback_reasons=reasons,
    )
