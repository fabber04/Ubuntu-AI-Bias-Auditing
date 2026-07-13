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

# World Bank World Development Indicators (WDI). Organized by the research
# domains this framework targets. All codes verified against api.worldbank.org
# — an indicator name alone doesn't guarantee the exact WDI code is right, so
# these were checked rather than guessed.
#
# Honest gap: WDI has no race/ethnicity variable for these countries. That's
# not an oversight — race isn't collected as a demographic category in African
# national statistics the way it is in, e.g., US Census data. The closest real
# proxies (ethnic fractionalization indices, e.g. Fearon 2003 or the ETH
# Zurich Ethnic Power Relations dataset) are academic one-time datasets, not
# live-API sources — worth a separate acquisition if this axis is needed,
# analogous to the PICES microdata route rather than a WDI pull.
WORLD_BANK_INDICATOR_CATEGORIES: Dict[str, Dict[str, str]] = {
    "Hiring / Labor Market": {
        "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
        "SL.UEM.TOTL.FE.ZS": "Unemployment, female (% of female labor force)",
        "SL.TLF.CACT.FE.ZS": "Labor force participation rate, female (% ages 15+)",
        "SL.TLF.CACT.MA.ZS": "Labor force participation rate, male (% ages 15+)",
        "SL.EMP.VULN.ZS": "Vulnerable employment, total (% of total employment) — informal-sector proxy",
    },
    "Finance": {
        "FX.OWN.TOTL.ZS": "Account ownership at a financial institution (% age 15+)",
        "FX.OWN.TOTL.FE.ZS": "Account ownership, female (% age 15+)",
        "FX.OWN.TOTL.MA.ZS": "Account ownership, male (% age 15+)",
        "FX.OWN.TOTL.PL.ZS": "Account ownership, poorest 40% (% age 15+)",
    },
    "Socio-Economic": {
        "SI.POV.NAHC": "Poverty headcount ratio at national poverty lines (% of population)",
        "SI.POV.GINI": "Gini index (income inequality)",
        "SE.ADT.LITR.ZS": "Literacy rate, adult total (% ages 15+)",
    },
    "Geographic / Infrastructure": {
        "SP.RUR.TOTL.ZS": "Rural population (% of total population)",
        "EG.ELC.ACCS.RU.ZS": "Access to electricity, rural (% of rural population)",
        "EG.ELC.ACCS.UR.ZS": "Access to electricity, urban (% of urban population)",
    },
    "Gender": {
        "SG.GEN.PARL.ZS": "Proportion of seats held by women in national parliaments (%)",
        "SE.ADT.LITR.FE.ZS": "Literacy rate, adult female (% ages 15+)",
    },
    "Health (existing)": {
        "SH.XPD.CHEX.GD.ZS": "Current health expenditure (% of GDP)",
    },
    "Digital Access (existing)": {
        "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    },
}

# Flattened lookup used by fetch_all_indicators / fetch_indicator — every
# category above collapses into one code->name dict for the API client.
WORLD_BANK_INDICATORS: Dict[str, str] = {
    code: name
    for category in WORLD_BANK_INDICATOR_CATEGORIES.values()
    for code, name in category.items()
}

# Sensible small default for the UI's initial selection — the full ~15
# indicators above are all available in the multiselect, but defaulting to
# all of them would make the first "Pull live data" click noticeably slower
# (each indicator x each country is a separate sequential API call).
WORLD_BANK_DEFAULT_INDICATORS: List[str] = [
    "IT.NET.USER.ZS",
    "FX.OWN.TOTL.ZS",
    "SL.UEM.TOTL.ZS",
    "SI.POV.GINI",
]

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
    0.3 in finance; see also the "hiring" preset below, weighted around the
    legal four-fifths/DIR standard). Defaults below are a neutral, equal-ish
    split; override per application domain via
    `FairnessWeights(w_dir=..., w_eod=..., w_dpd=...)`.
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
    # Hiring/recruitment: DIR gets the heaviest weight because the 0.80
    # threshold this framework already enforces (config.DIR_THRESHOLD) is
    # literally the "four-fifths rule" from the U.S. EEOC's Uniform Guidelines
    # on Employee Selection Procedures (1978) — the longest-standing legal
    # standard for adverse impact in hiring, and the closest thing to a
    # universally-recognized bright line these three metrics have. DPD carries
    # secondary weight as the raw-difference counterpart of the same
    # selection-rate concept (offer rates, interview-callback rates); EOD is
    # weighted lowest here not because it's unimportant, but because equalized
    # odds requires a ground-truth "qualified/unqualified" label that most
    # hiring pipelines don't cleanly have (unlike health outcomes or loan
    # repayment), making it noisier to act on in this domain specifically.
    "hiring": FairnessWeights(w_dir=0.5, w_eod=0.2, w_dpd=0.3),
    "neutral": FairnessWeights(w_dir=0.4, w_eod=0.3, w_dpd=0.3),
}

# w0 sweep: predictive-performance/fairness trade-off parameter (section 3.4/Tier 3)
W0_SWEEP_START = 0.0
W0_SWEEP_STOP = 1.0
W0_SWEEP_STEP = 0.05  # 21 candidate solutions, exactly as in the paper

# Minimum acceptable utility, expressed as a fraction of baseline utility.
UTILITY_RETENTION_FLOOR = 0.95  # gamma_utility = 95% of baseline (paper section 3.2/Tier 2)

RANDOM_SEED = 42