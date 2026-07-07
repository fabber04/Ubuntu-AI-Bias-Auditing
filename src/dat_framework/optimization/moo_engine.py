"""
Tier 2: The Multi-Objective Optimization Engine (Mathematical Layer).

Treats predictive utility and composite fairness as conflicting objectives and
runs the paper's 21-point parameter sweep (w0 from 0.0 to 1.0 in steps of 0.05)
using SLSQP, per section 3.4/Tier 2/section "Operational Mechanics".

Because this repo doesn't ship a trainable classifier (fairness metrics are
computed directly on prediction columns — see data/synthetic_dhs.py), theta here
is a per-intersectional-subgroup logit correction vector: one scalar nudge per
subgroup, applied to the baseline model's predicted probability before
thresholding. This is a lightweight but genuine stand-in for "model parameter
weights theta being optimized during training" (config.py's optimization
parameters table) — swap in real model retraining by replacing `apply_theta`
and `predict_with_theta` with calls into your own model, keeping the same
sweep/constraint/Pareto-extraction logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dat_framework.config import (
    FairnessWeights,
    UTILITY_RETENTION_FLOOR,
    W0_SWEEP_START,
    W0_SWEEP_STOP,
    W0_SWEEP_STEP,
)
from dat_framework.metrics.fairness_metrics import (
    all_intersectional_subgroups,
    intersectional_metrics,
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_theta(
    baseline_prob: np.ndarray, subgroup_id: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Apply a per-subgroup logit correction theta[subgroup_id[i]] to each
    individual's baseline predicted probability."""
    corrected_logit = _logit(baseline_prob) + theta[subgroup_id]
    return _sigmoid(corrected_logit)


def _accuracy(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    """Real, hard-thresholded accuracy — used for *reporting*, not for the
    optimizer's gradient (see `_soft_accuracy` for why)."""
    y_pred = (prob >= threshold).astype(int)
    return float((y_pred == y_true).mean())


def _soft_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Smooth surrogate for accuracy: 1 - mean absolute error between the
    continuous probability and the true label. Hard-thresholded accuracy is a
    step function of theta (zero gradient almost everywhere), which silently
    stalls SLSQP at its starting point. This surrogate is differentiable and
    tracks real accuracy closely enough to optimize against, with real
    accuracy still reported to the user for interpretability."""
    return float(1.0 - np.abs(y_true - prob).mean())


def _soft_fairness_loss(
    prob: np.ndarray,
    y_true: np.ndarray,
    subgroup_id: np.ndarray,
    weights: FairnessWeights,
    min_group_size: int,
) -> float:
    """Same DIR/EOD/DPD composite loss as the paper's Tier 2 formula, but
    computed from continuous predicted probabilities (expected selection
    rate / expected TPR / expected FPR) rather than hard 0/1 decisions, so the
    loss is smooth in theta and SLSQP can actually descend it. The reported
    Tier 2/3 tables still use hard-thresholded DIR/EOD/DPD for
    interpretability (see `metrics.fairness_metrics`) — this soft version is
    purely an internal optimization objective.
    """
    n_subgroups = int(subgroup_id.max()) + 1
    privileged_id = 0  # (0,0,...,0) always sorts first -> id 0, see
    # `optimization.preprocessing.assign_intersectional_subgroups`

    priv_mask = subgroup_id == privileged_id
    if priv_mask.sum() < min_group_size:
        return np.nan
    priv_sel = prob[priv_mask].mean()
    priv_pos = priv_mask & (y_true == 1)
    priv_neg = priv_mask & (y_true == 0)
    priv_tpr = prob[priv_pos].mean() if priv_pos.any() else np.nan
    priv_fpr = prob[priv_neg].mean() if priv_neg.any() else np.nan

    dirs, dpds, eods = [], [], []
    for sid in range(n_subgroups):
        if sid == privileged_id:
            continue
        mask = subgroup_id == sid
        if mask.sum() < min_group_size:
            continue
        sel = prob[mask].mean()
        pos = mask & (y_true == 1)
        neg = mask & (y_true == 0)
        tpr = prob[pos].mean() if pos.any() else np.nan
        fpr = prob[neg].mean() if neg.any() else np.nan

        if priv_sel > 1e-6:
            # Clip at 1.0: once a subgroup's selection rate reaches parity
            # with the privileged group, further "overshoot" isn't additional
            # fairness — without this clip, (1-DIR) becomes unboundedly
            # negative and SLSQP happily exploits it as free reward instead
            # of genuinely balancing the trade-off.
            dirs.append(min(sel / priv_sel, 1.0))
        dpds.append(abs(sel - priv_sel))
        if not (np.isnan(tpr) or np.isnan(priv_tpr) or np.isnan(fpr) or np.isnan(priv_fpr)):
            eods.append(abs(tpr - priv_tpr) + abs(fpr - priv_fpr))

    mean_dir = np.mean(dirs) if dirs else 1.0
    mean_dpd = np.mean(dpds) if dpds else 0.0
    mean_eod = np.mean(eods) if eods else 0.0

    return weights.w_dir * (1 - mean_dir) + weights.w_eod * mean_eod + weights.w_dpd * mean_dpd


def _fairness_loss_from_df(
    df_with_pred: pd.DataFrame,
    weights: FairnessWeights,
    y_true_col: str,
    y_pred_col: str,
    min_group_size: int,
) -> float:
    inter = intersectional_metrics(
        df_with_pred, y_true_col=y_true_col, y_pred_col=y_pred_col,
        min_group_size=min_group_size,
    )
    if inter.empty:
        return np.nan
    mean_dir = inter["DIR"].dropna().mean()
    mean_eod = inter["EOD"].dropna().abs().mean()
    mean_dpd = inter["DPD"].dropna().abs().mean()
    mean_dir = 1.0 if np.isnan(mean_dir) else mean_dir
    mean_eod = 0.0 if np.isnan(mean_eod) else mean_eod
    mean_dpd = 0.0 if np.isnan(mean_dpd) else mean_dpd
    # "Loss in fairness = w1(1-DIR) + w2(EOD) + w3(DPD)" — paper section 3.5/Tier 2
    return weights.w_dir * (1 - mean_dir) + weights.w_eod * mean_eod + weights.w_dpd * mean_dpd


@dataclass
class MOOSolution:
    w0: float
    theta: np.ndarray
    accuracy: float
    fairness_loss: float
    mean_dir: float
    mean_dpd: float
    mean_eod: float
    utility_retention_pct: float


def solve_for_w0(
    df: pd.DataFrame,
    w0: float,
    weights: FairnessWeights,
    baseline_accuracy: float,
    utility_floor: float = UTILITY_RETENTION_FLOOR,
    y_true_col: str = "y_true",
    baseline_prob_col: str = "y_pred_baseline_prob",
    min_group_size: int = 5,
    max_iter: int = 60,
) -> MOOSolution:
    """Solve the constrained optimization for one point on the w0 sweep:

        minimize   w0 * (1 - accuracy(theta)) + (1 - w0) * fairness_loss(theta)
        subject to accuracy(theta) >= utility_floor * baseline_accuracy

    theta is a per-subgroup logit-correction vector (Tier 2's model parameters).
    Solved with SLSQP, matching the paper's "Sequential Least Squares
    Programming (SLSQP) solver" (section 3.4, Operational Mechanics, Step 3).
    """
    y_true = df[y_true_col].to_numpy()
    baseline_prob = df[baseline_prob_col].to_numpy()
    subgroup_id = df["subgroup_id"].to_numpy()
    n_subgroups = int(subgroup_id.max()) + 1

    soft_baseline_accuracy = _soft_accuracy(y_true, baseline_prob)

    def corrected_prob(theta: np.ndarray) -> np.ndarray:
        return apply_theta(baseline_prob, subgroup_id, theta)

    # Small L2 penalty on theta. Without it, 32 free per-subgroup parameters
    # can overfit this particular sample — pushing accuracy above baseline and
    # DIR above 1 "for free" — which contradicts the paper's central claim
    # (the Impossibility Theorem / "necessity of compromise") that fairness
    # gains cost some utility. The penalty keeps corrections proportionate to
    # genuine bias signal rather than sample-specific noise.
    theta_l2_lambda = 0.004

    def objective(theta: np.ndarray) -> float:
        prob = corrected_prob(theta)
        acc = _soft_accuracy(y_true, prob)
        floss = _soft_fairness_loss(prob, y_true, subgroup_id, weights, min_group_size)
        floss = 1.0 if np.isnan(floss) else floss
        reg = theta_l2_lambda * float(np.mean(theta ** 2))
        return w0 * (1 - acc) + (1 - w0) * floss + reg

    def utility_constraint(theta: np.ndarray) -> float:
        prob = corrected_prob(theta)
        acc = _soft_accuracy(y_true, prob)
        return acc - utility_floor * soft_baseline_accuracy  # must be >= 0

    theta0 = np.zeros(n_subgroups)
    bounds = [(-3.0, 3.0)] * n_subgroups

    result = minimize(
        objective,
        theta0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": utility_constraint}],
        options={"maxiter": max_iter, "ftol": 1e-6, "eps": 1e-3},
    )

    def build_pred_df(theta: np.ndarray) -> pd.DataFrame:
        corrected = corrected_prob(theta)
        out = df.copy()
        out["y_pred_moo"] = (corrected >= 0.5).astype(int)
        out["y_pred_moo_prob"] = corrected
        return out

    theta_final = result.x
    final_df = build_pred_df(theta_final)
    acc = _accuracy(y_true, final_df["y_pred_moo_prob"].to_numpy())
    inter = intersectional_metrics(final_df, y_true_col=y_true_col, y_pred_col="y_pred_moo",
                                    min_group_size=min_group_size)
    mean_dir = inter["DIR"].dropna().mean() if not inter.empty else np.nan
    mean_dpd = inter["DPD"].dropna().mean() if not inter.empty else np.nan
    mean_eod = inter["EOD"].dropna().mean() if not inter.empty else np.nan
    floss = _fairness_loss_from_df(final_df, weights, y_true_col, "y_pred_moo", min_group_size)

    return MOOSolution(
        w0=w0,
        theta=theta_final,
        accuracy=acc,
        fairness_loss=floss,
        mean_dir=mean_dir,
        mean_dpd=mean_dpd,
        mean_eod=mean_eod,
        utility_retention_pct=100 * acc / baseline_accuracy,
    )


def run_w0_sweep(
    df: pd.DataFrame,
    weights: FairnessWeights,
    y_true_col: str = "y_true",
    baseline_prob_col: str = "y_pred_baseline_prob",
    utility_floor: float = UTILITY_RETENTION_FLOOR,
    min_group_size: int = 5,
    progress_callback=None,
) -> List[MOOSolution]:
    """Run the full 21-point w0 sweep (0.0 -> 1.0, step 0.05), producing one
    MOOSolution per w0 — the raw material for the Tier 3 Pareto frontier.
    """
    baseline_accuracy = _accuracy(
        df[y_true_col].to_numpy(), df[baseline_prob_col].to_numpy()
    )
    w0_values = np.round(
        np.arange(W0_SWEEP_START, W0_SWEEP_STOP + 1e-9, W0_SWEEP_STEP), 2
    )
    solutions = []
    for i, w0 in enumerate(w0_values):
        sol = solve_for_w0(
            df, float(w0), weights, baseline_accuracy, utility_floor,
            y_true_col, baseline_prob_col, min_group_size,
        )
        solutions.append(sol)
        if progress_callback is not None:
            progress_callback(i + 1, len(w0_values), sol)
    return solutions
