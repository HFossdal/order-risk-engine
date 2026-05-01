from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OrderRequest(BaseModel):
    """A single order to score.

    Field set mirrors the IEEE-CIS columns the model consumes. All fields are
    optional — XGBoost handles missingness — but real callers should provide
    as much as they have. `extra` is a passthrough for anything not enumerated
    here (e.g. less-common V/id columns) so the schema doesn't have to list
    339 V columns.

    Velocity fields (velocity_count_*, velocity_amt_sum_*, velocity_amt_mean_*)
    are computed offline in this codebase. In production the API would look
    them up from a feature store keyed on `card1`; for now callers can either
    pass them in directly or accept that they default to missing.
    """

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float | None = None
    ProductCD: str | None = None
    card1: int | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    TransactionDT: float | None = Field(
        default=None,
        description="Seconds from the dataset reference timestamp; used to derive hour/dayofweek.",
    )

    extra: dict[str, Any] = Field(default_factory=dict)


class ShapFeature(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: str


class PredictResponse(BaseModel):
    risk_score: float
    decision: str
    policy: dict
    top_shap_features: list[ShapFeature]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
