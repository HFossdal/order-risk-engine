"""Domain feature engineering on top of the merged IEEE-CIS table.

Tree models handle missingness and scale natively, so this module focuses on
producing features that aren't trivially derivable column-wise: velocity
counts/sums per card, transaction-amount log, and time-of-day signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CARD_KEY = "card1"
AMOUNT = "TransactionAmt"
TIME = "TransactionDT"

VELOCITY_WINDOWS = (5, 20, 100)


CORE_NUMERIC = [
    "TransactionAmt",
    "TransactionAmt_log",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D10", "D15",
    "hour", "dayofweek",
]

CORE_CATEGORICAL = [
    "ProductCD",
    "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_28", "id_29", "id_31", "id_35", "id_36", "id_37", "id_38",
]

VELOCITY_FEATURES = [
    f"velocity_{kind}_{w}"
    for w in VELOCITY_WINDOWS
    for kind in ("count", "amt_sum", "amt_mean")
]

FEATURE_GROUPS = {
    "numeric": CORE_NUMERIC + VELOCITY_FEATURES,
    "categorical": CORE_CATEGORICAL,
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TIME in out.columns:
        seconds = out[TIME].astype("float64")
        out["hour"] = ((seconds // 3600) % 24).astype("Int64")
        out["dayofweek"] = ((seconds // 86400) % 7).astype("Int64")
    if AMOUNT in out.columns:
        out["TransactionAmt_log"] = np.log1p(out[AMOUNT].clip(lower=0))
    return out


def add_velocity_features(
    df: pd.DataFrame,
    card_key: str = CARD_KEY,
    windows: tuple[int, ...] = VELOCITY_WINDOWS,
) -> pd.DataFrame:
    """Per-card rolling counts and amounts over the last N transactions.

    Uses transaction order (TransactionDT) rather than wall-clock windows so
    the feature matches what would be available at scoring time given a card's
    recent history buffer.
    """
    if card_key not in df.columns or TIME not in df.columns:
        return df

    out = df.sort_values(TIME, kind="mergesort").copy()
    grouped = out.groupby(card_key, sort=False)[AMOUNT]

    for w in windows:
        # shift(1) so the current transaction does not count itself
        rolled = grouped.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).count())
        out[f"velocity_count_{w}"] = rolled.astype("float32")

        rolled_sum = grouped.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum())
        out[f"velocity_amt_sum_{w}"] = rolled_sum.astype("float32")

        rolled_mean = grouped.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        out[f"velocity_amt_mean_{w}"] = rolled_mean.astype("float32")

    return out.sort_index()


def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature-engineering steps and return only the model columns
    (plus TransactionID and isFraud if present, for downstream splitting)."""
    df = add_time_features(df)
    df = add_velocity_features(df)

    keep = [c for c in FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"] if c in df.columns]
    extras = [c for c in ("TransactionID", "isFraud", TIME) if c in df.columns]

    missing = set(FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"]) - set(keep)
    for col in missing:
        df[col] = np.nan
    keep = FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"]

    return df[extras + keep]
