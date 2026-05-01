from model.decision import Decision, DecisionPolicy, threshold_sweep
import numpy as np
import pytest


def test_policy_routes_correctly():
    p = DecisionPolicy(approve_below=0.05, reject_above=0.65)
    assert p.decide(0.01) is Decision.APPROVE
    assert p.decide(0.04999) is Decision.APPROVE
    assert p.decide(0.05) is Decision.REVIEW
    assert p.decide(0.50) is Decision.REVIEW
    assert p.decide(0.65) is Decision.REJECT
    assert p.decide(0.99) is Decision.REJECT


def test_policy_validation():
    with pytest.raises(ValueError):
        DecisionPolicy(approve_below=0.9, reject_above=0.5)


def test_policy_roundtrip(tmp_path):
    p = DecisionPolicy(approve_below=0.3, reject_above=0.8)
    out = tmp_path / "policy.json"
    p.to_json(out)
    p2 = DecisionPolicy.from_json(out)
    assert p == p2


def test_decide_batch_routes_nan_to_review():
    p = DecisionPolicy(approve_below=0.10, reject_above=0.65)
    out = p.decide_batch([0.05, 0.5, 0.9, float("nan")])
    assert out == [Decision.APPROVE, Decision.REVIEW, Decision.REJECT, Decision.REVIEW]


def test_threshold_sweep_shape():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    rows = threshold_sweep(y_true, y_score, thresholds=[0.2, 0.5, 0.8])
    assert len(rows) == 3
    for r in rows:
        assert 0 <= r["precision"] <= 1
        assert 0 <= r["recall"] <= 1
