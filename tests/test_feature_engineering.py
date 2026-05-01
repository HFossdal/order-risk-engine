import numpy as np
import pandas as pd
from pipeline.feature_engineering import (
    add_time_features,
    add_velocity_features,
    assemble_features,
)


def _toy_frame():
    return pd.DataFrame(
        {
            "TransactionID": range(1, 9),
            "TransactionDT": [100, 200, 300, 400, 500, 600, 700, 800],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0],
            "card1": [1, 1, 1, 1, 2, 2, 2, 2],
            "isFraud": [0, 0, 1, 0, 0, 0, 0, 1],
        }
    )


def test_time_features_derive_hour_and_dow():
    df = add_time_features(_toy_frame())
    assert "hour" in df.columns
    assert "dayofweek" in df.columns
    assert "TransactionAmt_log" in df.columns


def test_velocity_features_only_use_history():
    df = add_velocity_features(_toy_frame())
    # First transaction for card1: no prior history, count_5 should be 0
    first_card1 = df[df["card1"] == 1].sort_values("TransactionDT").iloc[0]
    assert first_card1["velocity_count_5"] == 0
    # Second transaction for card1: one prior (the first), count_5 should be 1
    second_card1 = df[df["card1"] == 1].sort_values("TransactionDT").iloc[1]
    assert second_card1["velocity_count_5"] == 1


def test_assemble_features_returns_expected_columns():
    out = assemble_features(_toy_frame())
    assert "TransactionID" in out.columns
    assert "isFraud" in out.columns
    # Velocity columns are present even if input was tiny
    assert "velocity_count_5" in out.columns
    assert "velocity_amt_mean_20" in out.columns
