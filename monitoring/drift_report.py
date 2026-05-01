"""Generate an Evidently drift report.

Run with:
    python -m monitoring.drift_report

Strategy: take the validation slice that `model.train` saved as the reference,
synthesize a "current" slice with realistic drift (amount inflation, device-mix
shift, geography shift), then ask Evidently for a Data Drift + Target Drift +
Classification report. Output is an HTML file under monitoring/reports/.

In production, replace the synthetic drift with the actual production scoring
log over a recent time window, joined to ground-truth labels once they arrive.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    TargetDriftPreset,
)
from evidently.report import Report

from pipeline.feature_engineering import FEATURE_GROUPS

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model/artifacts"
OUT_DIR = ROOT / "monitoring/reports"


def _simulate_drift(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Realistic drift: amount distribution shifts up, device mix moves toward
    mobile, payer email domains get slightly different mix, and a fraction of
    velocity counts increase (mimicking a card-testing wave)."""
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    if "TransactionAmt" in drifted:
        drifted["TransactionAmt"] = drifted["TransactionAmt"] * rng.uniform(
            1.15, 1.45, size=len(drifted)
        )
        drifted["TransactionAmt_log"] = np.log1p(drifted["TransactionAmt"].clip(lower=0))

    if "DeviceType" in drifted:
        flip = rng.random(len(drifted)) < 0.25
        drifted.loc[flip, "DeviceType"] = "mobile"

    if "P_emaildomain" in drifted:
        flip = rng.random(len(drifted)) < 0.10
        drifted.loc[flip, "P_emaildomain"] = "anonymous.com"

    for w in (5, 20, 100):
        col = f"velocity_count_{w}"
        if col in drifted:
            spike = rng.random(len(drifted)) < 0.05
            drifted.loc[spike, col] = drifted.loc[spike, col].fillna(0) + rng.integers(
                3, 15, size=spike.sum()
            )

    return drifted


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_path = ARTIFACTS / "validation_sample.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"{sample_path} not found. Run `python -m model.train` first to produce it."
        )

    reference = pd.read_parquet(sample_path)
    current = _simulate_drift(reference)

    pipeline = joblib.load(ARTIFACTS / "calibrated_pipeline.joblib")
    feature_cols = FEATURE_GROUPS["numeric"] + FEATURE_GROUPS["categorical"]

    reference["prediction"] = pipeline.predict_proba(reference[feature_cols])[:, 1]
    current["prediction"] = pipeline.predict_proba(current[feature_cols])[:, 1]

    # Binarization threshold for the ClassificationPreset uses the deployed
    # auto-reject threshold rather than 0.5. Platt-compressed scores on
    # rare-positive data rarely cross 0.5, so 0.5 produces a degenerate
    # confusion matrix that doesn't reflect production behavior.
    from model.decision import DecisionPolicy
    policy_path = ARTIFACTS / "policy.json"
    bin_threshold = (
        DecisionPolicy.from_json(policy_path).reject_above
        if policy_path.exists()
        else 0.5
    )

    column_mapping = ColumnMapping(
        target="isFraud" if "isFraud" in reference.columns else None,
        prediction="prediction",
        numerical_features=[c for c in FEATURE_GROUPS["numeric"] if c in reference.columns],
        categorical_features=[c for c in FEATURE_GROUPS["categorical"] if c in reference.columns],
    )

    presets = [DataDriftPreset()]
    if column_mapping.target is not None:
        presets.append(TargetDriftPreset())
        # ClassificationPreset wants discrete predictions
        ref_pred = (reference["prediction"] >= bin_threshold).astype(int)
        cur_pred = (current["prediction"] >= bin_threshold).astype(int)
        ref_for_cls = reference.assign(prediction=ref_pred)
        cur_for_cls = current.assign(prediction=cur_pred)
        cls_report = Report(metrics=[ClassificationPreset()])
        cls_report.run(
            reference_data=ref_for_cls, current_data=cur_for_cls, column_mapping=column_mapping
        )
        cls_report.save_html(str(OUT_DIR / "classification_report.html"))

    drift_report = Report(metrics=presets)
    drift_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    out = OUT_DIR / "drift_report.html"
    drift_report.save_html(str(out))
    print(f"Wrote {out}")
    print(f"Classification report (if labels were present): {OUT_DIR / 'classification_report.html'}")


if __name__ == "__main__":
    main()
