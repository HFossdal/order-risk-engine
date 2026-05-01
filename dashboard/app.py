"""Streamlit operator dashboard.

Run locally:
    streamlit run dashboard/app.py

Two modes:
1. Direct in-process scoring (default) — loads the model artifacts directly.
2. API mode — POSTs to a running FastAPI service. Toggle with the sidebar.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from model.decision import DecisionPolicy
from model.explain import RiskExplainer
from pipeline.feature_engineering import (
    FEATURE_GROUPS,
    add_time_features,
)

ARTIFACTS = Path(os.environ.get("MODEL_ARTIFACTS", Path(__file__).resolve().parent.parent / "model/artifacts"))


PRESETS: dict[str, dict] = {
    "legit": {
        "label": "Typical legitimate order",
        "description": "Small daytime purchase on a long-established card. Should auto-approve.",
        "fields": {
            "amt": 89.50, "product_cd": "W", "card4": "visa", "card6": "credit",
            "card1": 13926, "card2": 583.0, "addr1": 315.0, "addr2": 87.0,
            "p_email": "gmail.com", "device_type": "desktop", "hour_of_day": 14,
            "c1": 1, "c13": 1, "d1": 250, "d15": 250,
        },
    },
    "card_tester": {
        "label": "Card-testing pattern",
        "description": "Many recent attempts on a brand-new card. The classic card-testing fraud signal — high C-counts, near-zero D values.",
        "fields": {
            "amt": 12.99, "product_cd": "C", "card4": "visa", "card6": "debit",
            "card1": 9500, "card2": 321.0, "addr1": 204.0, "addr2": 87.0,
            "p_email": "anonymous.com", "device_type": "mobile", "hour_of_day": 3,
            "c1": 18, "c13": 22, "d1": 0, "d15": 0,
        },
    },
    "stolen_card": {
        "label": "Suspected stolen card",
        "description": "Large unusual purchase, off-hours, anonymous email, mobile. Recent activity pattern consistent with a thief probing the card.",
        "fields": {
            "amt": 1899.99, "product_cd": "C", "card4": "visa", "card6": "credit",
            "card1": 9500, "card2": 321.0, "addr1": 204.0, "addr2": 87.0,
            "p_email": "anonymous.com", "device_type": "mobile", "hour_of_day": 3,
            "c1": 12, "c13": 12, "d1": 0, "d15": 0,
        },
    },
}


def _seed_form_defaults():
    """Initialize session_state with the legit preset's values on first load."""
    legit = PRESETS["legit"]["fields"]
    for k, v in legit.items():
        st.session_state.setdefault(k, v)


def _apply_preset(preset_key: str):
    for k, v in PRESETS[preset_key]["fields"].items():
        st.session_state[k] = v


@st.cache_resource(show_spinner="Loading model artifacts...")
def _load_local() -> RiskExplainer | None:
    try:
        return RiskExplainer.from_artifacts(ARTIFACTS)
    except FileNotFoundError:
        return None


def _load_policy() -> DecisionPolicy:
    p = ARTIFACTS / "policy.json"
    return DecisionPolicy.from_json(p) if p.exists() else DecisionPolicy()


def _load_metrics() -> dict | None:
    p = ARTIFACTS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _score_local(payload: dict, policy: DecisionPolicy):
    explainer = _load_local()
    if explainer is None:
        st.error("Model artifacts not found. Run `python -m model.train` first.")
        st.stop()
    row = pd.DataFrame([payload])
    row = add_time_features(row)
    for col in FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"]:
        if col not in row.columns:
            row[col] = None
    score = explainer.score(row)
    decision = policy.decide(score)
    contribs = explainer.explain(row, top_k=8)
    return score, decision.value, [
        {"feature": c.feature, "value": c.value, "shap_value": c.shap_value, "direction": c.direction}
        for c in contribs
    ]


def _score_via_api(api_url: str, payload: dict, policy: DecisionPolicy):
    """Fetch the score from the API but apply the dashboard's local policy.

    The API uses its own policy (loaded from policy.json at startup), but the
    operator console needs to reflect the sidebar sliders so ops can preview
    the effect of a threshold change without restarting the API.
    """
    r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    r.raise_for_status()
    body = r.json()
    score = body["risk_score"]
    decision = policy.decide(score).value
    return score, decision, body["top_shap_features"]


st.set_page_config(page_title="Order Risk Console", layout="wide")
st.title("Order Risk & Routing Console")
st.caption(
    "Score an order, see the calibrated risk probability, the routing "
    "decision, and the SHAP-driven explanation."
)


with st.sidebar:
    st.header("Settings")
    mode = st.radio("Scoring backend", ["In-process", "API"], index=0)
    api_url = st.text_input("API URL", value="http://localhost:8000")
    st.divider()
    st.subheader("Decision policy")
    base_policy = _load_policy()
    approve_below = st.slider(
        "auto-approve below", 0.0, 1.0, base_policy.approve_below, 0.001, format="%.3f"
    )
    reject_above = st.slider(
        "auto-reject above", 0.0, 1.0, base_policy.reject_above, 0.001, format="%.3f"
    )
    if approve_below > reject_above:
        st.error("approve_below must be ≤ reject_above")
        st.stop()
    policy = DecisionPolicy(approve_below=approve_below, reject_above=reject_above)


tab_score, tab_thresholds, tab_metrics = st.tabs(
    ["Score an order", "Threshold tradeoff", "Model metrics"]
)

with tab_score:
    _seed_form_defaults()

    st.subheader("Scenarios")
    st.caption(
        "One-click presets that load archetypal order patterns into the form below. "
        "Click one, then **Score order** to see how the model reasons about that pattern. "
        "Note: the model weights recent-history features (C/D) more heavily than amount or "
        "email — intuitive human fraud signals only fire when accompanied by anomalous activity patterns."
    )
    preset_cols = st.columns(3)
    for i, key in enumerate(("legit", "card_tester", "stolen_card")):
        with preset_cols[i]:
            preset = PRESETS[key]
            st.button(
                preset["label"],
                help=preset["description"],
                on_click=_apply_preset,
                args=(key,),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Order details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("Transaction amount", min_value=0.0, step=10.0, key="amt")
        st.selectbox("Product code", ["W", "C", "R", "H", "S"], key="product_cd")
        st.selectbox(
            "Card brand", ["visa", "mastercard", "american express", "discover"], key="card4"
        )
        st.selectbox("Card type", ["debit", "credit", "charge card"], key="card6")
    with col2:
        st.number_input("card1 (issuer code)", min_value=0, key="card1")
        st.number_input("card2", min_value=0.0, key="card2")
        st.number_input("addr1 (billing region)", min_value=0.0, key="addr1")
        st.number_input("addr2 (billing country)", min_value=0.0, key="addr2")
    with col3:
        st.selectbox(
            "Payer email domain",
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"],
            key="p_email",
        )
        st.selectbox("Device type", ["desktop", "mobile"], key="device_type")
        st.slider("Hour of day (UTC)", 0, 23, key="hour_of_day")

    with st.expander("Recent-history features (C/D — normally from a feature store)", expanded=False):
        st.caption(
            "Vesta's anonymized counts (C*) and timedeltas (D*) — most of the "
            "model's signal. Presets above set realistic values; tweak here to "
            "explore the boundary between bands."
        )
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.number_input("C1 (recent count)", min_value=0, step=1, key="c1")
        with h2:
            st.number_input("C13 (recent count)", min_value=0, step=1, key="c13")
        with h3:
            st.number_input("D1 (days since first seen)", min_value=0, step=1, key="d1")
        with h4:
            st.number_input("D15 (days since last seen)", min_value=0, step=1, key="d15")

    payload = {
        "TransactionAmt": st.session_state.amt,
        "ProductCD": st.session_state.product_cd,
        "card1": st.session_state.card1,
        "card2": st.session_state.card2,
        "card4": st.session_state.card4,
        "card6": st.session_state.card6,
        "addr1": st.session_state.addr1,
        "addr2": st.session_state.addr2,
        "P_emaildomain": st.session_state.p_email,
        "DeviceType": st.session_state.device_type,
        "TransactionDT": st.session_state.hour_of_day * 3600,
        "C1": st.session_state.c1,
        "C13": st.session_state.c13,
        "D1": st.session_state.d1,
        "D15": st.session_state.d15,
    }

    if st.button("Score order", type="primary"):
        if mode == "API":
            score, decision_str, shap_features = _score_via_api(api_url, payload, policy)
        else:
            score, decision_str, shap_features = _score_local(payload, policy)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Risk score", f"{score:.3f}")
            color = {"auto_approve": "green", "manual_review": "orange", "auto_reject": "red"}[
                decision_str
            ]
            st.markdown(f"### Decision: :{color}[{decision_str.replace('_', ' ').upper()}]")
            st.progress(min(score, 1.0))
        with c2:
            st.subheader("Top SHAP contributors")
            shap_df = pd.DataFrame(shap_features)
            if not shap_df.empty:
                shap_df = shap_df.sort_values("shap_value")
                fig = px.bar(
                    shap_df,
                    x="shap_value",
                    y="feature",
                    color="direction",
                    color_discrete_map={
                        "increases_risk": "#d62728",
                        "decreases_risk": "#2ca02c",
                    },
                    orientation="h",
                    height=320,
                )
                fig.update_layout(showlegend=True, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(shap_df, use_container_width=True, hide_index=True)


with tab_thresholds:
    st.subheader("Precision / recall tradeoff")
    metrics = _load_metrics()
    if not metrics:
        st.info("Train the model first (`python -m model.train`) to populate metrics.")
    else:
        sweep = pd.DataFrame(metrics["threshold_sweep"])
        st.dataframe(sweep, use_container_width=True, hide_index=True)
        fig = px.line(
            sweep.melt(
                id_vars="threshold",
                value_vars=["precision", "recall", "f1", "rejected_share"],
            ),
            x="threshold",
            y="value",
            color="variable",
            title="Metric vs. auto-reject threshold",
        )
        fig.add_vline(
            x=policy.reject_above, line_dash="dash", annotation_text="current reject_above"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_metrics:
    metrics = _load_metrics()
    if not metrics:
        st.info("No metrics on disk yet.")
    else:
        st.json(metrics["headline"])
