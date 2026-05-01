"""SHAP explanations for individual predictions.

CalibratedClassifierCV wraps the booster and SHAP doesn't natively support
that wrapper, so we run TreeExplainer against the *uncalibrated* booster.
Calibration is monotonic (Platt sigmoid), so the SHAP value ranking and
sign of each feature's contribution carry over to the calibrated score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import shap


@dataclass
class ShapContribution:
    feature: str
    value: float | str | None
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class RiskExplainer:
    """Loads the calibrated pipeline + the raw booster and explains predictions."""

    def __init__(
        self,
        calibrated_pipeline,
        preprocessor,
        booster,
        feature_names: Sequence[str],
    ):
        self.calibrated_pipeline = calibrated_pipeline
        self.preprocessor = preprocessor
        self.booster = booster
        self.feature_names = list(feature_names)
        self._explainer = shap.TreeExplainer(self.booster)

    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path) -> "RiskExplainer":
        d = Path(artifacts_dir)
        return cls(
            calibrated_pipeline=joblib.load(d / "calibrated_pipeline.joblib"),
            preprocessor=joblib.load(d / "preprocessor.joblib"),
            booster=joblib.load(d / "booster.joblib"),
            feature_names=joblib.load(d / "feature_names.joblib"),
        )

    def score(self, row: pd.DataFrame) -> float:
        proba = self.calibrated_pipeline.predict_proba(row)[:, 1]
        return float(proba[0])

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> list[ShapContribution]:
        X = self.preprocessor.transform(row)
        shap_values = self._explainer.shap_values(X)
        # binary: shap returns (n, n_features)
        contribs = np.asarray(shap_values)[0]

        order = np.argsort(np.abs(contribs))[::-1][:top_k]
        results: list[ShapContribution] = []
        for i in order:
            name = self.feature_names[i]
            raw_value = row.iloc[0].get(name, None)
            results.append(
                ShapContribution(
                    feature=name,
                    value=_jsonable(raw_value),
                    shap_value=float(contribs[i]),
                    direction="increases_risk" if contribs[i] > 0 else "decreases_risk",
                )
            )
        return results


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
