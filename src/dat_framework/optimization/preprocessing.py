"""
Tier 1: Sankofa-Inspired Intersectional Mapping (Diagnostic Layer).

Implements the null-space projection / linear adversarial debiasing described in
the DAT framework's Tier 1: strip latent proxies for protected attributes out of
the feature space via orthogonal projection, such that the mutual information
between the decontaminated features and the protected attributes falls below a
threshold epsilon (default 0.05), then assign every individual to their exact
intersectional subgroup via one-hot encoding (K = 2^m subgroups for m binarized
protected attributes).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from dat_framework.config import INFO_LEAKAGE_EPSILON, PROTECTED_ATTRIBUTES
from dat_framework.metrics.fairness_metrics import intersectional_group_labels


def null_space_project(
    X: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """Orthogonally project feature matrix X onto the null space of protected
    attribute matrix A, removing any component of X that is linearly explained
    by A. This is the "Null-Space Projection" named in Tier 1.

    X: (n_samples, n_features)
    A: (n_samples, n_protected_attrs) — should include an intercept column if
       you want to also remove mean-level dependence; we add one internally.
    """
    A_with_intercept = np.column_stack([np.ones(A.shape[0]), A])
    # Projection onto the column space of A: P_A = A (A^T A)^-1 A^T
    # Projection onto the null space of A: P_null = I - P_A
    gram = A_with_intercept.T @ A_with_intercept
    gram_inv = np.linalg.pinv(gram)  # pseudo-inverse for numerical stability
    P_A = A_with_intercept @ gram_inv @ A_with_intercept.T
    n = A_with_intercept.shape[0]
    P_null = np.eye(n) - P_A
    return P_null @ X


def measure_information_leakage(
    X_decontaminated: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """Mutual information I(X_j ; A) per decontaminated feature column j,
    against each protected attribute column, maxed across attributes to give
    a conservative per-feature leakage score. Values approaching zero indicate
    the statistical independence Tier 1 is aiming for (threshold epsilon)."""
    n_features = X_decontaminated.shape[1]
    n_attrs = A.shape[1]
    leakage = np.zeros((n_features, n_attrs))
    for j in range(n_attrs):
        mi = mutual_info_classif(
            X_decontaminated, A[:, j], discrete_features=False, random_state=0
        )
        leakage[:, j] = mi
    return leakage.max(axis=1)


def decontaminate_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    attribute_cols: List[str] = PROTECTED_ATTRIBUTES,
    epsilon: float = INFO_LEAKAGE_EPSILON,
) -> dict:
    """End-to-end Tier 1 pipeline: project out linear dependence on protected
    attributes, measure residual leakage, and flag features that still exceed
    the epsilon threshold (meaning a nonlinear proxy likely remains and those
    features may need to be dropped entirely rather than merely projected).
    """
    X = df[feature_cols].to_numpy(dtype=float)
    A = df[attribute_cols].to_numpy(dtype=float)

    X_decon = null_space_project(X, A)
    leakage = measure_information_leakage(X_decon, A)

    decon_df = pd.DataFrame(X_decon, columns=[f"{c}_decon" for c in feature_cols], index=df.index)
    leakage_report = pd.DataFrame(
        {"feature": feature_cols, "max_mutual_info_with_A": leakage,
         "exceeds_epsilon": leakage > epsilon}
    )

    return {"decontaminated_features": decon_df, "leakage_report": leakage_report}


def assign_intersectional_subgroups(
    df: pd.DataFrame, attributes: List[str] = PROTECTED_ATTRIBUTES
) -> pd.DataFrame:
    """Attach a `subgroup_tuple` and integer `subgroup_id` column (0..K-1) to
    df, implementing the "individual observations assigned to exact
    intersectional group via one-hot encoding" step of Tier 1.
    """
    out = df.copy()
    labels = intersectional_group_labels(out, attributes)
    unique_tuples = sorted(set(labels))
    tuple_to_id = {t: i for i, t in enumerate(unique_tuples)}
    out["subgroup_tuple"] = labels
    out["subgroup_id"] = labels.map(tuple_to_id)
    return out
