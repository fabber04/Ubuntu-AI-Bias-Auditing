"""
Central configuration for the DAT (Decolonial Appropriate Technology) framework.

All the "magic numbers" that appear in the paper live here, in one place, so that
re-running the pipeline with different countries / weights / thresholds is a one-line
change rather than a hunt through the codebase.
"""
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# 3.2 Data Collection — one country per Sub-Saharan sub-region
# ---------------------------------------------------------------------------
# ISO2 codes as used by the World Bank API.
SUB_SAHARAN_COUNTRIES: Dict[str, str] = {
    "GH": "Ghana",           # West Africa
    "ZW": "Zimbabwe",        # East / Southern Africa (paper's own case study)
    "KE": "Kenya",           # East Africa
    "CM": "Cameroon",        # Central Africa
    "ZA": "South Africa",    # Southern Africa
    "NG": "Nigeria",         # West Africa
}

# World Bank World Development Indicators used in the paper (section 3.2).
WORLD_BANK_INDICATORS: Dict[str, str] = {
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "FX.OWN.TOTL.ZS": "Borrowing from a financial institution (% age 15+)",
    "SH.XPD.CHEX.GD.ZS": "Current health expenditure (% of GDP)",
}

# ---------------------------------------------------------------------------
# 3.3 Phase I — Context-specific protected attributes (Okolo, 2022)
# ---------------------------------------------------------------------------
# Each attribute is binarized (paper section 3.1 of DAT framework, Tier 1) so that
# K = 2^len(PROTECTED_ATTRIBUTES) intersectional subgroups result. With 5 attributes
# below, K = 32, matching the paper's reported 32 subgroups.
PROTECTED_ATTRIBUTES: List[str] = [
    "rurality",          # rural (1) vs urban (0)
    "gender",             # female (1) vs male (0)
    "disability",         # disability (1) vs none (0)
    "socio_economic",     # low-income/informal (1) vs high-income/formal (0)
    "education",          # low-education (1) vs high-education (0)
]

# NOTE: the paper's own worked example uses 4 binarized attributes (K=16 in principle)
# but reports K=32 subgroups; we default to 5 attributes to land on K=32 exactly
# (2**5 == 32). Drop `education` from PROTECTED_ATTRIBUTES if you want to match the
# paper's narrower 4-attribute worked examples in section 6.2 instead (K=16).

# ---------------------------------------------------------------------------
# 3.4 Phase II — Fairness metric thresholds
# ---------------------------------------------------------------------------
DIR_THRESHOLD = 0.80   # DIR significantly below this => unacceptable bias
EOD_THRESHOLD = 0.10   # EOD above this => heavily biased system
DPD_THRESHOLD = 0.10   # DPD above this => bias toward dominant/favored group

# ---------------------------------------------------------------------------
# Tier 1 — Sankofa-inspired null-space projection
# ---------------------------------------------------------------------------
INFO_LEAKAGE_EPSILON = 0.05  # max allowable mutual information I(X;A)

# ---------------------------------------------------------------------------
# Tier 2 — Multi-objective optimization
# ---------------------------------------------------------------------------
@dataclass
class FairnessWeights:
    """Weighting coefficients w1, w2, w3 for DIR, EOD, DPD (must sum to 1).

    The paper notes these are domain-dependent (e.g. DIR weighted 0.7 in health,
    0.3 in finance). Defaults below are a neutral, equal-ish split; override per
    application domain via `FairnessWeights(w_dir=..., w_eod=..., w_dpd=...)`.
    """
    w_dir: float = 0.4
    w_eod: float = 0.3
    w_dpd: float = 0.3

    def __post_init__(self):
        total = self.w_dir + self.w_eod + self.w_dpd
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fairness weights must sum to 1.0, got {total}")


DOMAIN_WEIGHT_PRESETS: Dict[str, FairnessWeights] = {
    "health": FairnessWeights(w_dir=0.7, w_eod=0.15, w_dpd=0.15),
    "finance": FairnessWeights(w_dir=0.3, w_eod=0.35, w_dpd=0.35),
    "neutral": FairnessWeights(w_dir=0.4, w_eod=0.3, w_dpd=0.3),
}

# w0 sweep: predictive-performance/fairness trade-off parameter (section 3.4/Tier 3)
W0_SWEEP_START = 0.0
W0_SWEEP_STOP = 1.0
W0_SWEEP_STEP = 0.05  # 21 candidate solutions, exactly as in the paper

# Minimum acceptable utility, expressed as a fraction of baseline utility.
UTILITY_RETENTION_FLOOR = 0.95  # gamma_utility = 95% of baseline (paper section 3.2/Tier 2)

RANDOM_SEED = 42
