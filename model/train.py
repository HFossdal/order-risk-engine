"""Training entry point.

Run with:  python -m model.train

Splits the data chronologically (the public competition data is itself
time-ordered) into train / calibration / validation slices. XGBoost is fit
on the train slice, Platt scaling is fit on the calibration slice with
CalibratedClassifierCV(cv='prefit'), and the validation slice is used for
final metrics and threshold sweeps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pipeline import (
    add_time_features,
    add_velocity_features,
    build_preprocessor,
    load_raw,
    merge_transaction_identity,
)
from pipeline.feature_engineering import FEATURE_GROUPS
from model.decision import DEFAULT_POLICY, threshold_sweep

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TARGET = "isFraud"


def _chronological_split(df: pd.DataFrame, train_frac=0.6, cal_frac=0.2):
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + cal_frac))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def _select_xy(df: pd.DataFrame):
    feature_cols = FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"]
    return df[feature_cols], df[TARGET].astype(int)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/6] Loading raw CSVs...")
    transactions, identity = load_raw("train")
    df = merge_transaction_identity(transactions, identity)
    print(f"      merged shape: {df.shape}")

    print("[2/6] Engineering features...")
    df = add_time_features(df)
    df = add_velocity_features(df)

    train_df, cal_df, val_df = _chronological_split(df)
    print(f"      train={len(train_df)}  cal={len(cal_df)}  val={len(val_df)}")

    X_train, y_train = _select_xy(train_df)
    X_cal, y_cal = _select_xy(cal_df)
    X_val, y_val = _select_xy(val_df)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)
    print(f"      class balance: pos={pos} neg={neg} scale_pos_weight={scale_pos_weight:.2f}")

    print("[3/6] Fitting preprocessor + XGBoost...")
    preprocessor = build_preprocessor()
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    pipeline = Pipeline([("pre", preprocessor), ("clf", xgb)])
    pipeline.fit(X_train, y_train)

    print("[4/6] Calibrating with Platt scaling on held-out slice...")
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(X_cal, y_cal)

    print("[5/6] Evaluating on validation slice...")
    val_scores = calibrated.predict_proba(X_val)[:, 1]
    metrics = _compute_metrics(y_val.to_numpy(), val_scores)
    print(json.dumps(metrics["headline"], indent=2))

    print("[6/6] Persisting artifacts...")
    booster = pipeline.named_steps["clf"]
    joblib.dump(calibrated, ARTIFACTS / "calibrated_pipeline.joblib")
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    joblib.dump(booster, ARTIFACTS / "booster.joblib")
    joblib.dump(
        FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"],
        ARTIFACTS / "feature_names.joblib",
    )
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    DEFAULT_POLICY.to_json(ARTIFACTS / "policy.json")

    # Save a small reference dataset for the dashboard / drift monitor
    sample = val_df.sample(n=min(5000, len(val_df)), random_state=42)
    sample.to_parquet(ARTIFACTS / "validation_sample.parquet", index=False)

    print(f"Done in {time.time() - t0:.1f}s. Artifacts -> {ARTIFACTS}")


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # F1 at the default reject threshold
    pred_at_default = (y_score >= DEFAULT_POLICY.reject_above).astype(int)
    f1_default = f1_score(y_true, pred_at_default, zero_division=0)

    # Best F1 over the PR curve (informational)
    f1_curve = (2 * precision * recall) / np.clip(precision + recall, 1e-9, None)
    best_idx = int(np.argmax(f1_curve[:-1])) if len(thresholds) else 0
    best_f1 = float(f1_curve[best_idx])
    best_thr = float(thresholds[best_idx]) if len(thresholds) else 0.5

    return {
        "headline": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "f1_at_default_reject": float(f1_default),
            "best_f1": best_f1,
            "best_f1_threshold": best_thr,
        },
        "threshold_sweep": threshold_sweep(y_true, y_score),
    }


if __name__ == "__main__":
    main()
