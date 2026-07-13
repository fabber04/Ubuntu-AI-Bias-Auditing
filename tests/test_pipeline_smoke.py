"""Smoke tests — not exhaustive, just enough to catch obvious breakage."""
import numpy as np
import pandas as pd
import pytest

from dat_framework.config import PROTECTED_ATTRIBUTES, FairnessWeights
from dat_framework.data.synthetic_dhs import SyntheticDataConfig, generate_dataset
from dat_framework.metrics.fairness_metrics import (
    intersectional_metrics,
    single_attribute_metrics,
)
from dat_framework.metrics.stats_tests import factorial_anova, non_additivity_check, paired_ttests
from dat_framework.optimization.moo_engine import solve_for_w0, _accuracy
from dat_framework.optimization.pareto import extract_pareto_front, solutions_to_dataframe
from dat_framework.optimization.preprocessing import (
    assign_intersectional_subgroups,
    decontaminate_features,
)


@pytest.fixture(scope="module")
def df():
    data = generate_dataset(SyntheticDataConfig(n_records=1000, seed=1))
    return assign_intersectional_subgroups(data, PROTECTED_ATTRIBUTES)


def test_data_schema(df):
    for col in ["id", *PROTECTED_ATTRIBUTES, "qualification_score", "y_true",
                "y_pred_baseline", "y_pred_single_axis", "subgroup_id"]:
        assert col in df.columns
    assert df[PROTECTED_ATTRIBUTES].isin([0, 1]).all().all()


def test_single_attribute_metrics(df):
    result = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    assert set(result.index) == set(PROTECTED_ATTRIBUTES)
    assert result["DIR"].between(0, 3).all()  # sane range, not asserting exact paper values


def test_intersectional_metrics(df):
    result = intersectional_metrics(df, y_pred_col="y_pred_baseline", min_group_size=5)
    assert not result.empty
    assert "DIR" in result.columns


def test_paired_ttests(df):
    single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    inter = intersectional_metrics(df, y_pred_col="y_pred_baseline", min_group_size=5)
    result = paired_ttests(single, inter)
    assert set(result.index) == {"DIR", "DPD", "EOD"}


def test_factorial_anova_runs(df):
    table = factorial_anova(df, outcome_col="y_pred_baseline")
    assert "PR(>F)" in table.columns
    assert len(table) > len(PROTECTED_ATTRIBUTES)  # main effects + interactions + residual


def test_non_additivity_check(df):
    result = non_additivity_check(df, "rurality", "gender", outcome_col="y_pred_baseline")
    assert "compounding_gap" in result
    assert isinstance(result["is_super_additive_harm"], (bool, np.bool_))


def test_decontaminate_features(df):
    feature_cols = ["qualification_score"]
    result = decontaminate_features(df, feature_cols, PROTECTED_ATTRIBUTES)
    assert "decontaminated_features" in result
    assert "leakage_report" in result
    assert len(result["leakage_report"]) == len(feature_cols)


def test_moo_single_solve(df):
    baseline_acc = _accuracy(df["y_true"].to_numpy(), df["y_pred_baseline_prob"].to_numpy())
    sol = solve_for_w0(df, 0.5, FairnessWeights(), baseline_acc, min_group_size=5, max_iter=20)
    assert 0.0 <= sol.accuracy <= 1.0
    assert sol.theta.shape[0] == df["subgroup_id"].nunique() or sol.theta.shape[0] >= 1


def test_pareto_extraction():
    # Synthetic mini sweep: accuracy rises, fairness_loss rises too (typical trade-off)
    solutions_df = pd.DataFrame({
        "w0": [0.0, 0.5, 1.0],
        "mean_DIR": [0.9, 0.8, 0.6],
        "mean_DPD": [0.05, 0.1, 0.2],
        "mean_EOD": [0.05, 0.1, 0.2],
        "fairness_loss": [0.05, 0.15, 0.35],
        "accuracy_pct": [70, 78, 85],
        "utility_retention_pct": [90, 100, 108],
    })
    pareto = extract_pareto_front(solutions_df)
    assert pareto["is_pareto_optimal"].all()  # monotonic trade-off -> all non-dominated


def test_population_stability_index():
    from dat_framework.optimization.drift_monitoring import population_stability_index
    rng = np.random.default_rng(0)
    same_a = rng.normal(0, 1, 2000)
    same_b = rng.normal(0, 1, 2000)
    shifted = rng.normal(2.0, 1, 2000)
    assert population_stability_index(same_a, same_b) < 0.10
    assert population_stability_index(same_a, shifted) > 0.25


def test_drift_check_runs(df):
    from dat_framework.optimization.drift_monitoring import run_drift_check
    df2 = generate_dataset(SyntheticDataConfig(n_records=1000, seed=2))
    df2 = assign_intersectional_subgroups(df2, PROTECTED_ATTRIBUTES)
    report = run_drift_check(df, df2, feature_cols=["qualification_score"], min_group_size=5)
    assert isinstance(report.fallback_triggered, bool)


def test_audit_log_chain(tmp_path, df):
    from dat_framework.optimization.audit_log import AuditLog
    log = AuditLog(str(tmp_path / "audit.jsonl"))
    log.record(actor="tester", action="fairness_audit", dataset=df, decision_rationale="test entry")
    log.record(actor="tester", action="model_selection", dataset=df, decision_rationale="test 2", w0_selected=0.5)
    assert log.verify_chain()
    entries = log.read_all()
    assert len(entries) == 2