"""
Streamlit interface for the Decolonial Appropriate Technology (DAT) framework.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dat_framework.config import (
    DOMAIN_WEIGHT_PRESETS,
    FairnessWeights,
    PROTECTED_ATTRIBUTES,
    SUB_SAHARAN_COUNTRIES,
    UTILITY_RETENTION_FLOOR,
    WORLD_BANK_DEFAULT_INDICATORS,
    WORLD_BANK_INDICATOR_CATEGORIES,
    WORLD_BANK_INDICATORS,
)
from dat_framework.data.synthetic_dhs import SyntheticDataConfig, generate_dataset
from dat_framework.metrics.fairness_metrics import (
    intersectional_metrics,
    single_attribute_metrics,
)
from dat_framework.metrics.stats_tests import factorial_anova, non_additivity_check, paired_ttests
from dat_framework.optimization.moo_engine import run_w0_sweep
from dat_framework.optimization.pareto import (
    extract_pareto_front,
    recommend_solution,
    solutions_to_dataframe,
)
from dat_framework.optimization.preprocessing import assign_intersectional_subgroups

st.set_page_config(page_title="DAT Fairness Lab", page_icon="🌍", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "chart_accent" not in st.session_state:
    st.session_state.chart_accent = "Terracotta"

ACCENT_HEXES = {
    "Terracotta": {"light": "#C1552D", "dark": "#E58259"},
    "Forest Green": {"light": "#2F5233", "dark": "#5C9A73"},
    "Gold": {"light": "#B8842E", "dark": "#E8B85C"},
}


def current_accent_hex() -> str:
    mode = "dark" if st.session_state.dark_mode else "light"
    return ACCENT_HEXES[st.session_state.chart_accent][mode]


def inject_theme_css(dark: bool):
    if dark:
        vars_css = """
            --bg:#1E1A16; --card-bg:#27211B; --card-border:#3B3128;
            --text-primary:#F5EFE6; --text-secondary:#C9BBA8;
            --accent:#E07A4B; --accent-green:#5C9A73; --accent-gold:#E8B85C;
            --sidebar-bg:#221D18; --shadow:0 2px 10px rgba(0,0,0,0.45);
        """
    else:
        vars_css = """
            --bg:#FBF7F1; --card-bg:#FFFFFF; --card-border:#EDE1CE;
            --text-primary:#2B2420; --text-secondary:#6B5F52;
            --accent:#C1552D; --accent-green:#2F5233; --accent-gold:#B8842E;
            --sidebar-bg:#F4EEE1; --shadow:0 1px 4px rgba(43,36,32,0.10);
        """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {{ {vars_css} }}
        html, body, .stApp {{
            background-color: var(--bg) !important;
            font-family: 'Inter', sans-serif;
        }}
        .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown {{
            color: var(--text-primary);
        }}
        section[data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--card-border);
        }}
        div[data-testid="stExpander"] {{
            background-color: var(--card-bg);
            border: 1px solid var(--card-border) !important;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 6px; background-color: var(--card-bg);
            padding: 6px; border-radius: 12px; border: 1px solid var(--card-border);
        }}
        button[data-baseweb="tab"] {{
            border-radius: 8px !important; font-weight: 600;
            color: var(--text-secondary) !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background-color: var(--accent) !important; color: white !important;
        }}
        div.stButton > button {{
            border-radius: 8px; border: 1px solid var(--card-border); font-weight: 600;
        }}
        div.stButton > button[kind="primary"] {{
            background-color: var(--accent); border-color: var(--accent); color: white;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 10px; overflow: hidden; border: 1px solid var(--card-border);
        }}
        .dat-card {{
            background-color: var(--card-bg); border: 1px solid var(--card-border);
            border-radius: 14px; padding: 14px 18px; box-shadow: var(--shadow);
            height: 100%;
        }}
        .dat-card-icon {{ font-size: 1.3rem; margin-bottom: 2px; }}
        .dat-card-label {{
            color: var(--text-secondary); font-size: 0.75rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.03em;
        }}
        .dat-card-value {{
            color: var(--text-primary); font-size: 1.6rem; font-weight: 800;
            margin: 2px 0 2px 0; line-height: 1.1;
        }}
        .dat-card-sub {{ color: var(--text-secondary); font-size: 0.75rem; }}
        .dat-header-title {{
            font-size: 2rem; font-weight: 800; color: var(--text-primary);
            margin-bottom: 0; line-height: 1.15;
        }}
        .dat-header-sub {{ color: var(--text-secondary); font-size: 0.95rem; margin-top: 2px; }}
        .dat-header-byline {{ color: var(--text-secondary); font-size: 0.8rem; margin-top: 2px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme_css(st.session_state.dark_mode)


def metric_card_row(cards):
    cols = st.columns(len(cards))
    for col, (icon, label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="dat-card">
                    <div class="dat-card-icon">{icon}</div>
                    <div class="dat-card-label">{label}</div>
                    <div class="dat-card-value">{value}</div>
                    <div class="dat-card-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
        <span style="font-size:1.5rem;">🦁</span>
        <span style="font-size:1.15rem; font-weight:800;">DAT Fairness Lab</span>
    </div>
    <div style="color:var(--text-secondary); font-size:0.75rem; margin-bottom:14px;">v1.1.0</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar.expander("1. Dataset Configuration", expanded=True):
    n_records = st.slider("Number of synthetic records", 500, 6000, 2000, step=250)
    seed = st.number_input("Random seed", value=42, step=1)
    min_group_size = st.slider("Minimum subgroup size (for metric stability)", 3, 30, 8)

with st.sidebar.expander("2. Fairness Weights (Tier 2)", expanded=True):
    domain = st.selectbox(
        "Domain preset", ["neutral", "health", "finance", "hiring", "custom"],
        help="Paper section 3.5: 'the weight of DIR may be 0.7 in a health dataset, "
             "and 0.3 in a finance dataset.' Hiring is weighted around DIR since the "
             "0.80 threshold this tool enforces is the legal four-fifths rule "
             "(EEOC Uniform Guidelines, 1978).",
    )
    if domain == "custom":
        w_dir = st.slider("w1 — DIR weight", 0.0, 1.0, 0.4)
        w_eod = st.slider("w2 — EOD weight", 0.0, 1.0 - w_dir, min(0.3, 1 - w_dir))
        w_dpd = round(1.0 - w_dir - w_eod, 4)
        st.caption(f"w3 — DPD weight (auto): {w_dpd}")
        weights = FairnessWeights(w_dir=w_dir, w_eod=w_eod, w_dpd=w_dpd)
    else:
        weights = DOMAIN_WEIGHT_PRESETS[domain]
        st.caption(f"w1(DIR)={weights.w_dir}, w2(EOD)={weights.w_eod}, w3(DPD)={weights.w_dpd}")

with st.sidebar.expander("3. Model & Evaluation", expanded=False):
    utility_floor_pct = st.slider(
        "Utility floor (% of baseline accuracy)", 80, 100, int(UTILITY_RETENTION_FLOOR * 100),
        help="Tier 2's gamma_utility constraint: the MOO sweep won't accept a solution "
             "that drops accuracy below this percentage of the baseline model's accuracy.",
    )
    slsqp_max_iter = st.slider(
        "SLSQP max iterations", 20, 150, 60, step=10,
        help="More iterations can find a better-optimized point per w0, at the cost of "
             "a slower sweep. 60 is a reasonable default for datasets under ~3,000 records.",
    )

with st.sidebar.expander("4. Visualization Options", expanded=False):
    st.session_state.chart_accent = st.radio(
        "Chart accent color", list(ACCENT_HEXES.keys()),
        index=list(ACCENT_HEXES.keys()).index(st.session_state.chart_accent),
        horizontal=True,
    )
    st.caption("Applies to neutral (non-semantic) charts. Red/green violation coloring "
               "elsewhere is left alone since it carries meaning.")

st.sidebar.divider()
dark_toggle = st.sidebar.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
if dark_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_toggle
    st.rerun()

st.sidebar.caption(
    "Data note: DHS microdata is gated behind a registered-researcher account, so "
    "this app uses a structurally-equivalent **synthetic** stand-in calibrated to "
    "the paper's reported baseline bias magnitudes (tables 6.0/6.2). Swap in a real "
    "DHS extract by matching the schema in `data/synthetic_dhs.py`."
)


@st.cache_data(show_spinner=False)
def _get_data(n, seed_):
    df = generate_dataset(SyntheticDataConfig(n_records=n, seed=seed_))
    df = assign_intersectional_subgroups(df, PROTECTED_ATTRIBUTES)
    return df


df = _get_data(n_records, int(seed))

with st.sidebar.expander("5. Export & Reporting", expanded=False):
    st.download_button(
        "⬇ Export dataset (CSV)",
        data=df.drop(columns=["subgroup_tuple"]).to_csv(index=False).encode(),
        file_name="dat_synthetic_dataset.csv", mime="text/csv",
        use_container_width=True,
    )
    _report_single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    _report_lines = [
        "DAT Fairness Lab — Summary Report",
        f"Dataset: {len(df):,} synthetic records, seed={seed}, domain preset={domain}",
        "",
        "Single-attribute baseline DIR/DPD/EOD:",
        _report_single[["DIR", "DPD", "EOD"]].round(3).to_string(),
        "",
        f"Fairness weights: DIR={weights.w_dir}, EOD={weights.w_eod}, DPD={weights.w_dpd}",
        f"Minimum subgroup size: {min_group_size}",
    ]
    st.download_button(
        "⬇ Download report (TXT)",
        data="\n".join(_report_lines).encode(),
        file_name="dat_summary_report.txt", mime="text/plain",
        use_container_width=True,
    )

head_col, reset_col = st.columns([6, 1])
with head_col:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:14px;">
            <div style="font-size:2.1rem;">🌍</div>
            <div>
                <div class="dat-header-title">From Universalism to Ubuntu</div>
                <div class="dat-header-sub">Decolonial Appropriate Technology (DAT) Framework ·
                Intersectional fairness for AI/ML in Sub-Saharan Africa</div>
                <div class="dat-header-byline">Dube, Mguni &amp; Dube (2025) · ICAT 2026</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with reset_col:
    st.write("")
    if st.button("↻ Reset", use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

tabs = st.tabs([
    "📊 Data & Context",
    "🔍 Phase I–II: Bias Diagnostics",
    "📈 Statistical Validation",
    "⚖️ Tier 1–3: DAT / MOO Framework",
])

@st.cache_data(show_spinner=False)
def _get_worldbank_cached(countries: tuple, indicators: tuple):
    from dat_framework.data.worldbank import fetch_all_indicators, latest_value_per_country
    countries_dict = {k: SUB_SAHARAN_COUNTRIES[k] for k in countries}
    indicators_dict = {k: WORLD_BANK_INDICATORS[k] for k in indicators}
    raw = fetch_all_indicators(countries_dict, indicators_dict)
    return latest_value_per_country(raw)


with tabs[0]:
    positive_rate = df["y_true"].mean()
    metric_card_row([
        ("🗂️", "Dataset Size", f"{len(df):,}", "Synthetic Records"),
        ("👥", "Protected Attributes", f"{len(PROTECTED_ATTRIBUTES)}", "Intersectional Groups"),
        ("📈", "Positive Rate (Overall)", f"{positive_rate:.0%}", "y_true = 1"),
        ("⚖️", "Fairness Domain", domain.capitalize(), "Domain Preset"),
        ("🛡️", "Minimum Group Size", f"{min_group_size}", "For Metric Stability"),
    ])
    st.write("")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Synthetic individual-level dataset")
        st.dataframe(df.drop(columns=["subgroup_tuple"]).head(20), use_container_width=True)
        st.caption(f"{len(df):,} records · {len(PROTECTED_ATTRIBUTES)} protected attributes "
                   f"· {df['subgroup_id'].nunique()} intersectional subgroups populated "
                   f"(K = 2^{len(PROTECTED_ATTRIBUTES)} = {2**len(PROTECTED_ATTRIBUTES)} possible)")

    with col2:
        st.subheader("Attribute prevalence")
        prevalence = df[PROTECTED_ATTRIBUTES].mean().sort_values(ascending=True)
        fig = px.bar(
            prevalence, orientation="h",
            labels={"value": "Share of population marked disadvantaged", "index": ""},
            text=[f"{v:.0%}" for v in prevalence.values],
        )
        fig.update_traces(marker_color=current_accent_hex(), textposition="outside")
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("World Bank context (live pull)")
    st.caption(
        "Section 3.2's country selection — one per Sub-Saharan sub-region — pulled live "
        "from the World Bank's public WDI API. Pick any mix of indicators across domains "
        "below; note WDI has no race/ethnicity variable for these countries — that's a real "
        "gap in the data, not a missing checkbox here."
    )

    _indicator_to_category = {
        code: cat for cat, inds in WORLD_BANK_INDICATOR_CATEGORIES.items() for code in inds
    }

    wb_cols = st.columns([2, 3, 1])
    with wb_cols[0]:
        chosen_countries = st.multiselect(
            "Countries", options=list(SUB_SAHARAN_COUNTRIES.keys()),
            default=["ZW", "GH", "KE"],
            format_func=lambda c: f"{SUB_SAHARAN_COUNTRIES[c]} ({c})",
        )
    with wb_cols[1]:
        chosen_indicators = st.multiselect(
            "Indicators (any mix of domains)",
            options=list(WORLD_BANK_INDICATORS.keys()),
            default=WORLD_BANK_DEFAULT_INDICATORS,
            format_func=lambda i: f"[{_indicator_to_category[i]}] {WORLD_BANK_INDICATORS[i]}",
        )
    with wb_cols[2]:
        st.write("")
        st.write("")
        pull = st.button("Pull live data", use_container_width=True)

    if pull:
        if not chosen_countries or not chosen_indicators:
            st.warning("Pick at least one country and one indicator.")
        else:
            from dat_framework.data.worldbank import fetch_all_indicators, latest_value_per_country

            cache_key = (tuple(chosen_countries), tuple(chosen_indicators))
            if cache_key in st.session_state.get("_wb_cache", {}):
                wb_df = st.session_state["_wb_cache"][cache_key]
            else:
                countries_dict = {k: SUB_SAHARAN_COUNTRIES[k] for k in chosen_countries}
                indicators_dict = {k: WORLD_BANK_INDICATORS[k] for k in chosen_indicators}
                total_calls = len(countries_dict) * len(indicators_dict)
                progress_bar = st.progress(0.0, text=f"Starting... (0/{total_calls} calls)")

                def _wb_progress(done, total, label):
                    progress_bar.progress(done / total, text=f"Fetched: {label} ({done}/{total})")

                try:
                    raw = fetch_all_indicators(
                        countries_dict, indicators_dict, progress_callback=_wb_progress
                    )
                    wb_df = latest_value_per_country(raw)
                    progress_bar.empty()
                    st.session_state.setdefault("_wb_cache", {})[cache_key] = wb_df
                except Exception as exc:
                    progress_bar.empty()
                    st.error(
                        f"Couldn't reach the World Bank API from this environment ({exc}). "
                        "This needs outbound internet access to api.worldbank.org — check your "
                        "network settings if you're running this somewhere restricted."
                    )
                    wb_df = pd.DataFrame()

            if not wb_df.empty:
                pivot = wb_df.pivot(index="country", columns="indicator_name", values="value")
                st.dataframe(pivot, use_container_width=True)
                fig2 = px.bar(
                    wb_df, x="country", y="value", color="indicator_name", barmode="group",
                    labels={"value": "Latest reported value", "country": "", "indicator_name": ""},
                )
                st.plotly_chart(fig2, use_container_width=True)
            elif "_wb_cache" not in st.session_state or cache_key not in st.session_state.get("_wb_cache", {}):
                pass
            else:
                st.warning("World Bank API returned no data for this selection.")

with tabs[1]:
    st.subheader("6.0 / 6.1 — Single-attribute fairness metrics")
    st.caption("Comparing the biased baseline model against a classic single-axis fairness fix.")

    baseline_single = single_attribute_metrics(df, y_pred_col="y_pred_baseline")
    corrected_single = single_attribute_metrics(df, y_pred_col="y_pred_single_axis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Baseline (pre-fairness adjustment)**")
        st.dataframe(
            baseline_single[["DIR", "DPD", "EOD", "DIR_violation", "EOD_violation"]]
            .style.format({"DIR": "{:.2f}", "DPD": "{:.2f}", "EOD": "{:.2f}"}),
            use_container_width=True,
        )
    with c2:
        st.markdown("**After classic single-axis correction**")
        st.dataframe(
            corrected_single[["DIR", "DPD", "EOD", "DIR_violation", "EOD_violation"]]
            .style.format({"DIR": "{:.2f}", "DPD": "{:.2f}", "EOD": "{:.2f}"}),
            use_container_width=True,
        )

    st.info(
        "Notice DIR rises above 0.8 for every attribute after the single-axis fix — "
        "this is the paper's **'mirage of neutrality'**: fairness looks solved, "
        "one attribute at a time.",
        icon="🎭",
    )

    st.divider()
    st.subheader("6.2 — Intersectional subgroup metrics")
    st.caption(
        "The same 'corrected' model, now evaluated on intersectional subgroups "
        "(vs. the fully-privileged reference group). Watch DIR fall back below 0.8."
    )

    inter_corrected = intersectional_metrics(
        df, y_pred_col="y_pred_single_axis", min_group_size=min_group_size
    ).sort_values("DIR")

    if inter_corrected.empty:
        st.warning("No intersectional subgroups meet the minimum size threshold — lower it in the sidebar.")
    else:
        n_show = st.slider("Subgroups to show (most disadvantaged first)", 5, min(30, len(inter_corrected)), 10)
        show_df = inter_corrected.head(n_show)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=show_df.index, y=show_df["DIR"], name="DIR (post single-axis fix)",
            marker_color=["#d62728" if v < 0.8 else "#2ca02c" for v in show_df["DIR"]],
        ))
        fig3.add_hline(y=0.8, line_dash="dash", line_color="gray",
                        annotation_text="0.80 disparate-impact threshold")
        fig3.update_layout(height=420, xaxis_tickangle=-35,
                            yaxis_title="Disparate Impact Ratio", xaxis_title="Intersectional subgroup",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(
            show_df[["DIR", "DPD", "EOD", "n_marginalized", "DIR_violation"]]
            .style.format({"DIR": "{:.2f}", "DPD": "{:.2f}", "EOD": "{:.2f}"}),
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("t-tests: single-attribute vs. intersectional")
    baseline_inter = intersectional_metrics(
        df, y_pred_col="y_pred_baseline", min_group_size=min_group_size
    )
    if baseline_inter.empty:
        st.warning("No intersectional subgroups meet the minimum size threshold — lower it in the sidebar.")
    else:
        ttest_df = paired_ttests(baseline_single, baseline_inter)
        st.dataframe(
            ttest_df.style.format({
                "mean_single": "{:.3f}", "mean_intersectional": "{:.3f}",
                "t_statistic": "{:.2f}", "p_value": "{:.4f}",
            }),
            use_container_width=True,
        )
        sig = ttest_df[ttest_df["p_value"] < 0.05]
        if not sig.empty:
            st.success(
                f"{len(sig)}/3 metrics show a statistically significant gap (p<0.05) between "
                "single-attribute and intersectional bias levels.",
                icon="✅",
            )

    st.divider()
    st.subheader("Factorial ANOVA — is intersectional harm non-additive?")
    st.caption(
        "A significant interaction term means two attributes' joint effect on the "
        "prediction isn't just the sum of their individual effects — the mathematical "
        "signature of compounding, non-additive intersectional harm."
    )
    anova_table = factorial_anova(df, outcome_col="y_pred_baseline")
    interaction_rows = anova_table[anova_table.index.str.contains(":")]
    main_rows = anova_table[~anova_table.index.str.contains(":")]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Main effects**")
        st.dataframe(main_rows.style.format({"sum_sq": "{:.2f}", "F": "{:.2f}", "PR(>F)": "{:.2e}"}),
                     use_container_width=True)
    with c2:
        st.markdown("**Pairwise interaction effects**")
        sig_interactions = (interaction_rows["PR(>F)"] < 0.05).sum()
        st.dataframe(interaction_rows.style.format({"sum_sq": "{:.2f}", "F": "{:.2f}", "PR(>F)": "{:.2e}"}),
                     use_container_width=True)
        st.caption(f"{sig_interactions}/{len(interaction_rows)} pairwise interactions significant at p<0.05.")

    st.divider()
    st.subheader("Non-additivity spotlight: pick two attributes")
    a1, a2 = st.columns(2)
    with a1:
        attr_a = st.selectbox("Attribute A", PROTECTED_ATTRIBUTES, index=0)
    with a2:
        attr_b = st.selectbox("Attribute B", PROTECTED_ATTRIBUTES, index=1)

    if attr_a != attr_b:
        check = non_additivity_check(df, attr_a, attr_b, outcome_col="y_pred_baseline")
        fig4 = go.Figure()
        labels = ["Neither", f"Only {attr_a}", f"Only {attr_b}", "Both (observed)", "Both (additive prediction)"]
        values = [
            check["reference_rate"], check[f"only_{attr_a}_rate"], check[f"only_{attr_b}_rate"],
            check["observed_joint_rate"], check["additive_prediction"],
        ]
        colors = ["#9c9c9c", current_accent_hex(), current_accent_hex(), "#d62728", "#e8a23d"]
        fig4.add_trace(go.Bar(x=labels, y=values, marker_color=colors))
        fig4.update_layout(height=380, yaxis_title="Mean positive prediction rate",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)
        gap = check["compounding_gap"]
        if check["is_super_additive_harm"]:
            st.warning(
                f"Compounding gap = {gap:.3f}: being **both** {attr_a} and {attr_b} disadvantaged "
                f"hurts more than the sum of each penalty alone — non-additive, super-linear harm.",
                icon="⚠️",
            )
        else:
            st.info(f"Compounding gap = {gap:.3f}: no super-additive harm detected for this pair.")
    else:
        st.caption("Pick two different attributes to compare.")

with tabs[3]:
    st.subheader("Tier 1 — Sankofa-inspired intersectional mapping")
    st.caption(
        f"{df['subgroup_id'].nunique()} of {2**len(PROTECTED_ATTRIBUTES)} possible intersectional "
        f"subgroups (K = 2^{len(PROTECTED_ATTRIBUTES)}) are populated in this dataset, each assigned "
        "via one-hot encoding of the protected-attribute tuple."
    )

    st.divider()
    st.subheader("Tier 2 — Multi-objective optimization sweep")
    st.caption(
        "Runs the paper's 21-point w₀ parameter sweep (0.0 → 1.0, step 0.05) via SLSQP, "
        "trading predictive utility against the composite fairness loss "
        "w1(1-DIR) + w2·EOD + w3·DPD."
    )

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_moo = st.button("▶ Run MOO sweep", type="primary", use_container_width=True)
    with info_col:
        st.caption(
            f"Weights: DIR={weights.w_dir}, EOD={weights.w_eod}, DPD={weights.w_dpd} · "
            f"min subgroup size={min_group_size} · utility floor={utility_floor_pct}% · "
            f"max SLSQP iterations={slsqp_max_iter} · this takes roughly "
            f"{max(1, n_records // 40)}s–{max(2, n_records // 15)}s depending on dataset size."
        )

    if run_moo:
        progress = st.progress(0.0, text="Starting sweep...")

        def _cb(i, total, sol):
            progress.progress(i / total, text=f"w0={sol.w0:.2f} ({i}/{total}) — "
                                               f"accuracy={sol.accuracy:.2%}, fairness_loss={sol.fairness_loss:.3f}")

        with st.spinner("Running 21-point SLSQP sweep..."):
            solutions = run_w0_sweep(
                df, weights, min_group_size=min_group_size,
                utility_floor=utility_floor_pct / 100, max_iter=slsqp_max_iter,
                progress_callback=_cb,
            )
        progress.empty()
        st.session_state["moo_solutions_df"] = solutions_to_dataframe(solutions)
        st.success("Sweep complete.")

    if "moo_solutions_df" in st.session_state:
        solutions_df = st.session_state["moo_solutions_df"]
        pareto_df = extract_pareto_front(solutions_df)

        st.divider()
        st.subheader("Tier 3 — Ubuntu-based Pareto frontier & governance layer")
        st.caption(
            "Per the paper: *'final model selection cannot be designated to automated "
            "computation.'* Explore the frontier below and pick the trade-off you're "
            "comfortable with — a human decision, not an automated one."
        )

        fig5 = go.Figure()
        non_pareto = pareto_df[~pareto_df["is_pareto_optimal"]]
        pareto = pareto_df[pareto_df["is_pareto_optimal"]]
        fig5.add_trace(go.Scatter(
            x=non_pareto["fairness_loss"], y=non_pareto["accuracy_pct"],
            mode="markers", name="Dominated solutions",
            marker=dict(color="lightgray", size=9),
            text=[f"w0={w:.2f}" for w in non_pareto["w0"]],
        ))
        fig5.add_trace(go.Scatter(
            x=pareto["fairness_loss"], y=pareto["accuracy_pct"],
            mode="markers+lines", name="Pareto-optimal front",
            marker=dict(color=current_accent_hex(), size=12, symbol="diamond"),
            line=dict(color=current_accent_hex()),
            text=[f"w0={w:.2f}" for w in pareto["w0"]],
        ))
        fig5.update_layout(
            xaxis_title="Fairness loss (lower = fairer)",
            yaxis_title="Predictive accuracy (%)",
            height=460,
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### Walk the frontier yourself")
        w0_choice = st.select_slider(
            "w₀ — utility/fairness trade-off (0 = prioritize fairness, 1 = prioritize utility)",
            options=list(np.round(solutions_df["w0"].to_numpy(), 2)),
            value=float(recommend_solution(pareto_df, priority="balanced")["w0"]),
        )
        chosen = solutions_df[np.isclose(solutions_df["w0"], w0_choice)].iloc[0]
        metric_card_row([
            ("🎯", "Accuracy", f"{chosen['accuracy_pct']:.1f}%", f"w0={w0_choice:.2f}"),
            ("⚖️", "Mean DIR", f"{chosen['mean_DIR']:.2f}", "Disparate Impact Ratio"),
            ("🛡️", "Utility Retention", f"{chosen['utility_retention_pct']:.1f}%", "vs. baseline"),
            ("📉", "Fairness Loss", f"{chosen['fairness_loss']:.3f}", "Lower is fairer"),
        ])

        recommended = recommend_solution(pareto_df, utility_floor_pct=float(utility_floor_pct), priority="balanced")
        st.caption(
            f"💡 A balanced starting point (≥{utility_floor_pct}% utility retention, lowest fairness loss): "
            f"**w₀ = {recommended['w0']:.2f}** — accuracy {recommended['accuracy_pct']:.1f}%, "
            f"mean DIR {recommended['mean_DIR']:.2f}. This is a suggestion to anchor discussion, "
            "not the final word — that's the governance layer's job, not the optimizer's."
        )

        with st.expander("Full sweep table"):
            st.dataframe(
                pareto_df.style.format({
                    "mean_DIR": "{:.3f}", "mean_DPD": "{:.3f}", "mean_EOD": "{:.3f}",
                    "fairness_loss": "{:.3f}", "accuracy_pct": "{:.2f}",
                    "utility_retention_pct": "{:.1f}",
                }),
                use_container_width=True,
            )
    else:
        st.info("Click **Run MOO sweep** above to populate the Pareto frontier.", icon="👆")

st.markdown(
    """
    <div style="text-align:center; color:var(--text-secondary); font-size:0.78rem; margin-top:2rem;">
        Built for research. Designed for impact. &nbsp;|&nbsp; DAT Fairness Lab © 2026
    </div>
    """,
    unsafe_allow_html=True,
)