"""Decision policy on top of calibrated risk scores.

Thresholds intentionally live outside the model so risk operations can tune
them without retraining. The training script writes a default policy to
`model/artifacts/policy.json`; ops can edit that file or pass a custom policy
into `decide`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np


class Decision(str, Enum):
    APPROVE = "auto_approve"
    REVIEW = "manual_review"
    REJECT = "auto_reject"


@dataclass(frozen=True)
class DecisionPolicy:
    # Defaults tuned to the calibrated score distribution observed on the
    # IEEE-CIS validation slice. Platt scaling compresses scores on highly
    # imbalanced data — the original "reject above 0.85" default never fired.
    # At reject_above=0.65 the model blocks ~1.3% of orders at ~78% precision
    # and ~29% recall on the held-out slice. Below approve_below=0.05 the
    # observed fraud rate is ~1.25% (vs the 3.4% base rate), so those orders
    # can safely auto-approve. Re-tune via threshold_sweep on your own data
    # before production.
    approve_below: float = 0.05
    reject_above: float = 0.65

    def __post_init__(self):
        if not 0.0 <= self.approve_below <= self.reject_above <= 1.0:
            raise ValueError(
                f"Invalid policy: approve_below={self.approve_below} "
                f"reject_above={self.reject_above}"
            )

    def decide(self, score: float) -> Decision:
        if score >= self.reject_above:
            return Decision.REJECT
        if score < self.approve_below:
            return Decision.APPROVE
        return Decision.REVIEW

    def decide_batch(self, scores: Iterable[float]) -> list[Decision]:
        """Vectorised version of `decide`.

        NaN scores are routed to manual review (both the `>=` and `<`
        comparisons return False for NaN, so they fall through to the
        REVIEW default). This matches the conservative stance of `decide`.
        """
        s = np.asarray(list(scores), dtype="float64")
        out = np.full(s.shape, Decision.REVIEW.value, dtype=object)
        out[s >= self.reject_above] = Decision.REJECT.value
        out[s < self.approve_below] = Decision.APPROVE.value
        return [Decision(v) for v in out]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionPolicy":
        return cls(**json.loads(Path(path).read_text()))


DEFAULT_POLICY = DecisionPolicy()


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> list[dict]:
    """Precision/recall/volume tradeoff for the auto-reject threshold.

    Useful to plug into Streamlit / a notebook for picking `reject_above`.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "rejected_share": float(pred.mean()),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return rows
