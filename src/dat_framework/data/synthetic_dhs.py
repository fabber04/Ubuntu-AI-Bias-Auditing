"""
Synthetic stand-in for DHS (Demographic and Health Surveys) individual-level records.

The DHS Program gates its microdata behind a registered-researcher account and
per-survey project approval (see paper section 3.2) — nothing in this repo can
fetch that for you. This module instead generates a *structurally equivalent*
dataset: same protected attributes, same rough compounding-bias story the paper
reports in tables 6.0-6.2, so the rest of the pipeline (fairness metrics, ANOVA,
MOO) is fully runnable today.

Swap this out for a real DHS extract the moment you have Program access: just
produce a DataFrame with the columns listed in `SCHEMA` below and everything
downstream (fairness_metrics, stats_tests, optimization.*) will work unmodified.

SCHEMA
------
id                 : unique respondent id
rurality           : 1 = rural, 0 = urban
gender             : 1 = female, 0 = male
disability         : 1 = has a disability, 0 = does not
socio_economic     : 1 = low-income / informal sector, 0 = high-income / formal
education          : 1 = low educational attainment, 0 = high
qualification_score: latent, attribute-blind "merit" signal in [0, 1]
                      (e.g. credit-worthiness / health-need score built only from
                      non-protected features — the ground truth a *fair* model
                      should track)
y_true             : ground-truth positive outcome (1 = should receive the
                      resource/service), drawn from qualification_score alone
y_pred_baseline    : biased model prediction (no fairness intervention)
y_pred_single_axis : prediction after classic, single-attribute fairness
                      correction only (reweighing per attribute independently)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dat_framework.config import PROTECTED_ATTRIBUTES, RANDOM_SEED


@dataclass
class SyntheticDataConfig:
    n_records: int = 4000
    seed: int = RANDOM_SEED
    # Marginal probability that an individual carries the "disadvantaged" side
    # of each attribute (rough Sub-Saharan Africa DHS-informed priors).
    p_rural: float = 0.62
    p_female: float = 0.51
    p_disability: float = 0.12
    p_low_income: float = 0.55
    p_low_education: float = 0.48
    # Non-additive bias injection coefficients: penalty(k) = alpha*k + gamma*k^2
    # where k = number of disadvantaged protected attributes an individual holds.
    # The quadratic term is what makes intersectional harm > sum of parts.
    # Calibrated so single-attribute baseline DIR lands ~0.73-0.80 and deep
    # intersectional subgroups fall to ~0.2-0.6, tracking the paper's tables
    # 6.0 (single-attribute) and 6.2 (intersectional) magnitudes.
    alpha_linear_penalty: float = 0.03
    gamma_quadratic_penalty: float = 0.011
    noise_std: float = 0.12


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(cfg: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """Generate a synthetic DHS-like individual-level dataset.

    The generative story:
    1. Each attribute in `config.PROTECTED_ATTRIBUTES` is sampled as an
       independent Bernoulli (with mild realistic correlation added below),
       matching the binarization the paper's Tier 1 layer expects.
    2. A protected-attribute-blind `qualification_score` is sampled — this is
       the "merit" signal a genuinely fair model should key off of.
    3. `y_true` is a noisy draw from `qualification_score` alone (fair ground
       truth: outcomes should NOT depend on protected attributes).
    4. `y_pred_baseline` reproduces algorithmic bias by *subtracting* a penalty
       from the qualification score before thresholding — the penalty grows
       super-linearly (alpha*k + gamma*k^2) in the count k of disadvantaged
       attributes an individual holds, which is exactly the "penalty of being
       a rural woman is greater than the sum of its parts" effect the paper's
       ANOVA is built to detect.
    5. `y_pred_single_axis` applies a classic, independent per-attribute
       reweighing correction (each attribute's *linear* penalty term is
       cancelled), leaving the quadratic/compounding term untouched — this
       reproduces the paper's "mirage of neutrality" finding (Table 6.1):
       single-axis metrics look fixed, intersectional groups are still hurt.
    """
    cfg = cfg or SyntheticDataConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_records

    # --- 1. Protected attributes with mild realistic correlation ---------
    rurality = rng.binomial(1, cfg.p_rural, n)
    # rural residents somewhat more likely to be low-income/low-education
    p_low_income_adj = np.clip(cfg.p_low_income + 0.15 * rurality, 0, 1)
    p_low_edu_adj = np.clip(cfg.p_low_education + 0.12 * rurality, 0, 1)

    gender = rng.binomial(1, cfg.p_female, n)
    disability = rng.binomial(1, cfg.p_disability, n)
    socio_economic = rng.binomial(1, p_low_income_adj)
    education = rng.binomial(1, p_low_edu_adj)

    attrs = {
        "rurality": rurality,
        "gender": gender,
        "disability": disability,
        "socio_economic": socio_economic,
        "education": education,
    }
    # keep only the attributes configured centrally, in case someone trims the list
    attrs = {k: v for k, v in attrs.items() if k in PROTECTED_ATTRIBUTES}
    k_disadvantaged = np.sum(np.vstack(list(attrs.values())), axis=0)

    # --- 2. Attribute-blind qualification / merit score -------------------
    qualification_score = np.clip(
        rng.normal(loc=0.55, scale=0.18, size=n), 0.0, 1.0
    )

    # --- 3. Ground truth: fair outcome depends only on qualification ------
    y_true_prob = _sigmoid(6 * (qualification_score - 0.5))
    y_true = rng.binomial(1, y_true_prob)

    # --- 4. Biased baseline model prediction ------------------------------
    penalty = (
        cfg.alpha_linear_penalty * k_disadvantaged
        + cfg.gamma_quadratic_penalty * k_disadvantaged ** 2
    )
    noise = rng.normal(0, cfg.noise_std, n)
    baseline_logit = 6 * (qualification_score - 0.5) - 6 * penalty + noise
    y_pred_baseline_prob = _sigmoid(baseline_logit)
    y_pred_baseline = rng.binomial(1, y_pred_baseline_prob)

    # --- 5. Single-axis fairness correction (linear term only) ------------
    # Reweighing per attribute independently cancels each attribute's average
    # linear contribution but cannot see the k^2 cross-term -> compounding
    # harm survives for intersectional subgroups.
    linear_only_penalty = cfg.alpha_linear_penalty * k_disadvantaged
    corrected_logit = (
        6 * (qualification_score - 0.5) - 6 * (penalty - linear_only_penalty) + noise
    )
    y_pred_single_axis_prob = _sigmoid(corrected_logit)
    y_pred_single_axis = rng.binomial(1, y_pred_single_axis_prob)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            **attrs,
            "qualification_score": qualification_score,
            "y_true": y_true,
            "y_pred_baseline": y_pred_baseline,
            "y_pred_baseline_prob": y_pred_baseline_prob,
            "y_pred_single_axis": y_pred_single_axis,
            "y_pred_single_axis_prob": y_pred_single_axis_prob,
        }
    )
    return df


if __name__ == "__main__":
    data = generate_dataset()
    out_path = "data/raw/synthetic_dhs.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved {len(data)} synthetic records to {out_path}")
    print(data.head())
