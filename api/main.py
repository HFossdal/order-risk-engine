"""FastAPI service exposing the calibrated risk model.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, OrderRequest, PredictResponse, ShapFeature
from model.decision import DecisionPolicy
from model.explain import RiskExplainer
from pipeline.feature_engineering import (
    FEATURE_GROUPS,
    add_time_features,
    add_velocity_features,
)

ARTIFACTS = Path(os.environ.get("MODEL_ARTIFACTS", Path(__file__).resolve().parent.parent / "model/artifacts"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v0.1")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["explainer"] = RiskExplainer.from_artifacts(ARTIFACTS)
        policy_path = Path(ARTIFACTS) / "policy.json"
        state["policy"] = (
            DecisionPolicy.from_json(policy_path) if policy_path.exists() else DecisionPolicy()
        )
        state["loaded"] = True
    except FileNotFoundError as e:
        state["loaded"] = False
        state["load_error"] = str(e)
    yield


app = FastAPI(
    title="Intelligent Order Risk & Routing API",
    description=(
        "Scores e-commerce orders for fraud risk and routes them to "
        "auto-approve / manual review / auto-reject based on a calibrated "
        "XGBoost model with SHAP explanations."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state.get("loaded") else "degraded",
        model_loaded=bool(state.get("loaded")),
        model_version=MODEL_VERSION if state.get("loaded") else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(order: OrderRequest):
    if not state.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not loaded: {state.get('load_error', 'unknown')}",
        )

    explainer: RiskExplainer = state["explainer"]
    policy: DecisionPolicy = state["policy"]

    row = _build_input_row(order)

    score = explainer.score(row)
    decision = policy.decide(score)
    contributions = explainer.explain(row, top_k=5)

    return PredictResponse(
        risk_score=score,
        decision=decision.value,
        policy={"approve_below": policy.approve_below, "reject_above": policy.reject_above},
        top_shap_features=[
            ShapFeature(
                feature=c.feature,
                value=c.value,
                shap_value=c.shap_value,
                direction=c.direction,
            )
            for c in contributions
        ],
        model_version=MODEL_VERSION,
    )


def _build_input_row(order: OrderRequest) -> pd.DataFrame:
    """Coerce the request into a 1-row DataFrame with all expected columns.

    Missing columns become NaN (XGBoost handles them). `extra` lets callers
    pass less-common fields (e.g. V123, id_31) without us listing all 400+ in
    the schema."""
    payload = order.model_dump(exclude={"extra"}, exclude_none=False)
    payload.update(order.extra or {})

    row = pd.DataFrame([payload])
    row = add_time_features(row)
    # add_velocity_features needs a population to roll over; for a single
    # request we only fill columns that the caller already provided in `extra`.
    for col in FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"]:
        if col not in row.columns:
            row[col] = None
    return row
