"""
Statistical validation from the paper's section 3.5:

1. Paired t-tests comparing mean DIR/DPD/EOD between single-attribute results
   and intersectional-subgroup results (paper table 6.3).
2. A factorial ANOVA testing whether intersectional bias is non-additive, i.e.
   whether the interaction term between protected attributes is significant
   above and beyond their main effects — this is the paper's central empirical
   claim (F(1,94)=18.4, p<0.001 in the original study).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from dat_framework.config import PROTECTED_ATTRIBUTES


def paired_ttests(
    single_attribute_df: pd.DataFrame, intersectional_df: pd.DataFrame
) -> pd.DataFrame:
    """Independent-samples t-test comparing single-attribute vs intersectional
    metric distributions, for each of DIR/DPD/EOD. Reproduces paper table 6.3.
    """
    rows = []
    for metric in ["DIR", "DPD", "EOD"]:
        single_vals = single_attribute_df[metric].dropna().to_numpy()
        inter_vals = intersectional_df[metric].dropna().to_numpy()
        if len(single_vals) < 2 or len(inter_vals) < 2:
            rows.append(
                {"metric": metric, "mean_single": np.nan, "mean_intersectional": np.nan,
                 "t_statistic": np.nan, "p_value": np.nan}
            )
            continue
        t_stat, p_val = stats.ttest_ind(single_vals, inter_vals, equal_var=False)
        rows.append(
            {
                "metric": metric,
                "mean_single": single_vals.mean(),
                "mean_intersectional": inter_vals.mean(),
                "t_statistic": t_stat,
                "p_value": p_val,
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def factorial_anova(
    df: pd.DataFrame,
    outcome_col: str = "y_pred_baseline",
    attributes: List[str] = PROTECTED_ATTRIBUTES,
    max_interaction_order: int = 2,
) -> pd.DataFrame:
    """Factorial ANOVA on the prediction outcome, with main effects for every
    protected attribute plus pairwise interaction terms.

    A significant interaction term is the mathematical signature of non-additive
    intersectional harm: it means the joint effect of two attributes on the
    outcome is not simply the sum of their individual effects, which is exactly
    what the paper uses ANOVA to demonstrate (section 3.5 / 4.2 / 6.3).
    """
    work = df.copy()
    # statsmodels' formula API is happiest with clean column names
    safe_attrs = attributes
    main_terms = " + ".join(f"C({a})" for a in safe_attrs)

    interaction_terms = []
    if max_interaction_order >= 2:
        for i in range(len(safe_attrs)):
            for j in range(i + 1, len(safe_attrs)):
                interaction_terms.append(f"C({safe_attrs[i]}):C({safe_attrs[j]})")

    formula_terms = [main_terms] + interaction_terms
    formula = f"{outcome_col} ~ " + " + ".join(formula_terms)

    model = ols(formula, data=work).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def non_additivity_check(
    df: pd.DataFrame,
    attr_a: str,
    attr_b: str,
    outcome_col: str = "y_pred_baseline",
) -> dict:
    """Directly checks: penalty(a AND b) vs penalty(a) + penalty(b).

    Returns the mean outcome for the reference group (neither attribute), each
    single-attribute group, and the joint intersectional group, plus the
    "additive prediction" (what you'd expect if harms simply summed) vs the
    actual observed joint effect — the gap between them is the non-additive
    / compounding component the paper's ANOVA formalizes.
    """
    ref = df[(df[attr_a] == 0) & (df[attr_b] == 0)][outcome_col].mean()
    only_a = df[(df[attr_a] == 1) & (df[attr_b] == 0)][outcome_col].mean()
    only_b = df[(df[attr_a] == 0) & (df[attr_b] == 1)][outcome_col].mean()
    both = df[(df[attr_a] == 1) & (df[attr_b] == 1)][outcome_col].mean()

    effect_a = ref - only_a
    effect_b = ref - only_b
    additive_prediction = ref - (effect_a + effect_b)
    observed = both
    compounding_gap = additive_prediction - observed  # positive => harm is super-additive

    return {
        "reference_rate": ref,
        f"only_{attr_a}_rate": only_a,
        f"only_{attr_b}_rate": only_b,
        "both_rate": both,
        "additive_prediction": additive_prediction,
        "observed_joint_rate": observed,
        "compounding_gap": compounding_gap,
        "is_super_additive_harm": compounding_gap > 0,
    }
