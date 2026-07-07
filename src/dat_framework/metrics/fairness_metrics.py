"""
Fairness metrics from the paper's section 3.4 (Phase II):

- Disparate Impact Ratio (DIR)          = P(Y_hat=1 | A=0) / P(Y_hat=1 | A=1)
- Equalized Odds Difference (EOD)       = |TPR(A=0)-TPR(A=1)| + |FPR(A=0)-FPR(A=1)|
- Demographic Parity Difference (DPD)   = |P(Y_hat=1|A=0) - P(Y_hat=1|A=1)|

Convention (matches the paper): A=0 is the marginalized subgroup, A=1 the
privileged/reference group. All three metrics are computed both per single
protected attribute and per intersectional subgroup (one-hot Cartesian product
of every attribute in `config.PROTECTED_ATTRIBUTES`).
"""
from __future__ import annotations

from itertools import product
from typing import Iterable, List

import numpy as np
import pandas as pd

from dat_framework.config import (
    DIR_THRESHOLD,
    DPD_THRESHOLD,
    EOD_THRESHOLD,
    PROTECTED_ATTRIBUTES,
)


def _rates(df: pd.DataFrame, mask: pd.Series, y_true_col: str, y_pred_col: str):
    """Return (selection_rate, TPR, FPR) for the subset of df where mask is True."""
    sub = df.loc[mask]
    if len(sub) == 0:
        return np.nan, np.nan, np.nan

    y_true = sub[y_true_col].to_numpy()
    y_pred = sub[y_pred_col].to_numpy()

    selection_rate = y_pred.mean()

    positives = y_true == 1
    negatives = y_true == 0
    tpr = y_pred[positives].mean() if positives.any() else np.nan
    fpr = y_pred[negatives].mean() if negatives.any() else np.nan
    return selection_rate, tpr, fpr


def compute_group_metrics(
    df: pd.DataFrame,
    group_mask_marginalized: pd.Series,
    group_mask_privileged: pd.Series,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred_baseline",
) -> dict:
    """Compute DIR, DPD, EOD comparing a marginalized group (A=0) to a privileged
    reference group (A=1), for one prediction column."""
    sel0, tpr0, fpr0 = _rates(df, group_mask_marginalized, y_true_col, y_pred_col)
    sel1, tpr1, fpr1 = _rates(df, group_mask_privileged, y_true_col, y_pred_col)

    dir_ = sel0 / sel1 if sel1 not in (0, np.nan) and not np.isnan(sel1) else np.nan
    dpd = sel0 - sel1
    eod = (
        (abs(tpr0 - tpr1) if not (np.isnan(tpr0) or np.isnan(tpr1)) else np.nan)
        + (abs(fpr0 - fpr1) if not (np.isnan(fpr0) or np.isnan(fpr1)) else np.nan)
    )

    return {
        "n_marginalized": int(group_mask_marginalized.sum()),
        "n_privileged": int(group_mask_privileged.sum()),
        "selection_rate_marginalized": sel0,
        "selection_rate_privileged": sel1,
        "DIR": dir_,
        "DPD": dpd,
        "EOD": eod,
        "DIR_violation": (not np.isnan(dir_)) and dir_ < DIR_THRESHOLD,
        "DPD_violation": (not np.isnan(dpd)) and abs(dpd) > DPD_THRESHOLD,
        "EOD_violation": (not np.isnan(eod)) and abs(eod) > EOD_THRESHOLD,
    }


def single_attribute_metrics(
    df: pd.DataFrame,
    attributes: Iterable[str] = PROTECTED_ATTRIBUTES,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred_baseline",
) -> pd.DataFrame:
    """Reproduces paper tables 6.0/6.1: one row per protected attribute."""
    rows = []
    for attr in attributes:
        marginalized = df[attr] == 1
        privileged = df[attr] == 0
        metrics = compute_group_metrics(df, marginalized, privileged, y_true_col, y_pred_col)
        rows.append({"attribute": attr, **metrics})
    return pd.DataFrame(rows).set_index("attribute")


def intersectional_group_labels(
    df: pd.DataFrame, attributes: List[str] = PROTECTED_ATTRIBUTES
) -> pd.Series:
    """One-hot Cartesian product assignment: each row -> its exact subgroup tuple,
    e.g. (1, 1, 0, 0, 0) for a rural woman with no disability, high income, high edu.
    Matches Tier 1's "Individual observations are then assigned to their exact
    intersectional group using one-hot encoding" (section 3.1 of the DAT framework).
    """
    return df[attributes].apply(lambda row: tuple(row.values), axis=1)


def all_intersectional_subgroups(attributes: List[str] = PROTECTED_ATTRIBUTES):
    """All 2^len(attributes) possible subgroup tuples (K=32 for 5 attributes)."""
    return list(product([0, 1], repeat=len(attributes)))


def intersectional_metrics(
    df: pd.DataFrame,
    attributes: List[str] = PROTECTED_ATTRIBUTES,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred_baseline",
    min_group_size: int = 5,
    privileged_tuple: tuple | None = None,
) -> pd.DataFrame:
    """Reproduces paper table 6.2: one row per intersectional subgroup, each
    compared against the fully-privileged reference group (all attributes = 0)
    by default.
    """
    labels = intersectional_group_labels(df, attributes)
    if privileged_tuple is None:
        privileged_tuple = tuple([0] * len(attributes))
    privileged_mask = labels.apply(lambda t: t == privileged_tuple)

    rows = []
    for subgroup in all_intersectional_subgroups(attributes):
        mask = labels.apply(lambda t: t == subgroup)
        if mask.sum() < min_group_size or subgroup == privileged_tuple:
            continue
        metrics = compute_group_metrics(df, mask, privileged_mask, y_true_col, y_pred_col)
        label = ", ".join(a for a, v in zip(attributes, subgroup) if v == 1) or "none"
        rows.append({"subgroup": label, "subgroup_tuple": subgroup, **metrics})

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("subgroup")
    return result


def summarize_bias_report(
    df: pd.DataFrame,
    attributes: List[str] = PROTECTED_ATTRIBUTES,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred_baseline",
    min_group_size: int = 5,
) -> dict:
    """Convenience wrapper returning both single-attribute and intersectional
    tables for one prediction column, e.g. baseline vs single-axis-corrected."""
    return {
        "single_attribute": single_attribute_metrics(df, attributes, y_true_col, y_pred_col),
        "intersectional": intersectional_metrics(
            df, attributes, y_true_col, y_pred_col, min_group_size
        ),
    }
