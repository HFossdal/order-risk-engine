# Intelligent Order Risk & Routing System

A production-shaped fraud detection and decision-automation service. Each
inbound order is scored by a calibrated XGBoost model, routed to one of
**auto-approve / manual review / auto-reject**, and accompanied by a
SHAP-based explanation suitable for analysts and chargeback evidence.

Trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset (~590k transactions, ~3.5% fraud).

---

## The business problem

Card-not-present fraud is asymmetric:

| Outcome | What it costs |
|---|---|
| **False negative** (approve a fraud) | Full transaction value + chargeback fee (typ. $15–$50) + scheme penalty if your fraud rate trips a threshold. Direct P&L hit. |
| **False positive** (reject a good order) | Lost revenue on the order **plus** the lifetime value of a customer who hits a "your card was declined" page. Industry benchmarks put the LTV cost at 5–10× the order value. |
| **Manual review** | Analyst time (~$2–$5 per case). Cheap relative to either error, **if** your queue is sized correctly. |

A single point estimate (e.g. "score > 0.5 → block") collapses this nuance.
Risk teams need:

1. A **calibrated probability** they can reason about — a 0.7 should mean
   "70% of orders like this are fraudulent," not "this came out higher than
   the other one."
2. **Configurable thresholds** so ops can re-tune as the fraud mix shifts
   without retraining.
3. A **review band** so the model can punt low-confidence cases to humans
   instead of guessing.
4. A **reason** for every decision — for analyst efficiency, for chargeback
   representment, and for regulatory defensibility.

This service provides all four.

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              Streamlit Console              │
                 │   (operator UI: score, tune, inspect SHAP)  │
                 └───────────────┬─────────────────────────────┘
                                 │ HTTP (or in-process)
                                 ▼
   ┌───────────────────┐   ┌───────────────────────────────┐   ┌──────────────────────┐
   │  Order intake     │──▶│       FastAPI service         │──▶│  Decision sink       │
   │  (checkout, etc.) │   │  POST /predict                │   │  (PSP / queue / OMS) │
   └───────────────────┘   │   ├─ feature assembly         │   └──────────────────────┘
                           │   ├─ calibrated XGBoost       │
                           │   ├─ threshold policy         │
                           │   └─ SHAP explainer (tree)    │
                           └─────┬───────────────────────┬─┘
                                 │ artifacts (joblib)    │ scores + labels
                                 ▼                       ▼
                         ┌───────────────┐       ┌──────────────────┐
                         │ model/        │       │ Evidently        │
                         │ artifacts/    │       │ drift report     │
                         │ • booster     │       │ (data + target + │
                         │ • calibrator  │       │  classification) │
                         │ • policy.json │       └──────────────────┘
                         └───────────────┘
```

Repo layout:

```
order-risk-engine/
├── data/                   # CSVs go here (gitignored)
├── pipeline/               # data loading + feature engineering + sklearn ColumnTransformer
├── model/
│   ├── train.py            # end-to-end training + calibration
│   ├── decision.py         # DecisionPolicy + threshold sweep
│   ├── explain.py          # SHAP TreeExplainer wrapper
│   └── artifacts/          # joblib + json outputs
├── api/                    # FastAPI app
├── dashboard/              # Streamlit operator console
├── monitoring/             # Evidently drift report
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Get the data

Download the IEEE-CIS competition data and unzip it into `data/`:

```bash
# requires `kaggle` CLI configured
kaggle competitions download -c ieee-fraud-detection -p data/
unzip -o data/ieee-fraud-detection.zip -d data/
```

Or download manually from
<https://www.kaggle.com/competitions/ieee-fraud-detection/data> and place
`train_transaction.csv` and `train_identity.csv` in `data/`. See
`data/README.md` for the schema.

### 2. Train

```bash
python -m model.train
```

This:
- merges transaction + identity tables on `TransactionID`
- engineers velocity features (rolling count/sum/mean per card over the
  last 5/20/100 transactions)
- splits chronologically (60/20/20 train/calibration/validation) — random
  splits leak future information on this dataset
- fits XGBoost with `scale_pos_weight = neg/pos`
- wraps with `CalibratedClassifierCV(method="sigmoid", cv="prefit")` for
  Platt scaling on the held-out calibration slice
- writes:
  - `model/artifacts/calibrated_pipeline.joblib`
  - `model/artifacts/booster.joblib`  (raw booster for SHAP)
  - `model/artifacts/preprocessor.joblib`
  - `model/artifacts/feature_names.joblib`
  - `model/artifacts/metrics.json`     (ROC-AUC, PR-AUC, F1, threshold sweep)
  - `model/artifacts/policy.json`      (default thresholds)
  - `model/artifacts/validation_sample.parquet` (reference for drift monitor)

Performance on the IEEE-CIS held-out chronological validation slice
(118k rows, ~3.4% positives):

- ROC-AUC: **0.874**
- PR-AUC: **0.450**  (≈13× the no-skill baseline of 0.034)
- F1 at default `reject_above=0.65`: **0.420**
- Brier score: **0.023** (naive baseline 0.032 — calibration adds real lift)

At the default reject threshold the model blocks ~1.3% of orders at
~78% precision and ~29% recall. The full precision/recall/F1 sweep across
thresholds 0.05–0.95 is in `metrics.json` and visible in the dashboard's
**Threshold tradeoff** tab.

(Numbers will vary slightly with random seed and XGBoost version.)

### 3. Serve

```bash
uvicorn api.main:app --reload --port 8000
```

Or with Docker:

```bash
docker compose up --build
# API:        http://localhost:8000/docs
# Dashboard:  http://localhost:8501
```

Sample request:

```bash
python scripts/sample_payload.py | \
    curl -s -X POST http://localhost:8000/predict \
        -H 'Content-Type: application/json' -d @- | jq
```

Response:

```json
{
  "risk_score": 0.0404,
  "decision": "auto_approve",
  "policy": { "approve_below": 0.05, "reject_above": 0.65 },
  "top_shap_features": [
    { "feature": "C13", "value": 18,  "shap_value": +0.92, "direction": "increases_risk" },
    { "feature": "C1",  "value": 12,  "shap_value": +0.74, "direction": "increases_risk" },
    { "feature": "D1",  "value": 0,   "shap_value": -0.51, "direction": "decreases_risk" },
    { "feature": "C14", "value": 14,  "shap_value": +0.43, "direction": "increases_risk" },
    { "feature": "TransactionAmt", "value": 1899.99, "shap_value": +0.18, "direction": "increases_risk" }
  ],
  "model_version": "v0.1"
}
```

Top SHAP contributors are typically Vesta's anonymized count/timedelta
columns (`C*`, `D*`); the human-meaningful fields rarely crack the top 5.
See **Feature-store dependency** below for why.

### 4. Operator dashboard

```bash
streamlit run dashboard/app.py
```

Lets you submit a sample order, see the score and decision, visualise SHAP
contributions, and re-tune thresholds against the saved precision/recall
sweep without retraining.

### 5. Drift monitoring

```bash
python -m monitoring.drift_report
open monitoring/reports/drift_report.html
```

Generates a synthetic-drift comparison (amount inflation, device-mix shift,
card-testing-style velocity spikes) against the held-out validation slice
using Evidently's data-drift, target-drift, and classification presets.

### 6. Tests

```bash
pytest -q
```

The included tests don't require trained artifacts — they cover the
decision policy and feature-engineering invariants.

---

## Decision policy

Default thresholds (in `model/artifacts/policy.json`):

| Score band     | Decision        |
|----------------|-----------------|
| `≥ 0.65`       | `auto_reject`   |
| `0.05 – 0.65`  | `manual_review` |
| `< 0.05`       | `auto_approve`  |

> **Why these numbers, not 0.40 / 0.85?** Platt scaling on the IEEE-CIS
> data (~3.4% positives) compresses calibrated scores — the maximum score
> on the held-out validation slice is ~0.76, so a 0.85 reject threshold
> never fires. The defaults above were chosen against the actual
> validation sweep: `reject_above=0.65` blocks ~1.3% of orders at ~78%
> precision and ~29% recall, and below `approve_below=0.05` the observed
> fraud rate is ~1.25% — confidently low enough to auto-approve.

`metrics.json` contains a precision/recall/F1 sweep across `reject_above`
from 0.05 to 0.95. The dashboard's **Threshold tradeoff** tab lets ops walk
that curve interactively and pick a setting based on the current cost
balance (chargeback rate, manual-review queue capacity, FP tolerance).

Thresholds are stored separately from the model so risk operations can
re-tune them without a retraining cycle.

## ⚠ Feature-store dependency

Most of the discriminative signal in the IEEE-CIS dataset comes from
**Vesta's anonymized count / timedelta columns** (`C1..C14`, `D1..D15`)
and the engineered velocity features. SHAP confirms it: the top-5
contributing features for almost every prediction are `C13`, `C1`,
`D1`, and the velocity counts.

That has a direct consequence for the API contract:

- An API call that supplies only the **human-meaningful fields**
  (TransactionAmt, ProductCD, card4, P_emaildomain, DeviceType, …) will
  score nearly every order around the base rate — the model has no
  recent-history signal to differentiate them.
- A useful production caller has to supply, or the API has to look up,
  per-card recent-history features (counts on the card in the last
  N hours, time since last transaction, etc.) keyed by `card1` /
  card hash.

This repo ships a sample payload (`scripts/sample_payload.py`) that
includes representative `C*`/`D*` values so you can see the model
actually discriminate. In a real deployment those values come from a
streaming feature store (Feast, Tecton, or a homegrown Redis + Flink
setup) — see the **Production scale** section below.

---

## What changes at production scale

This repo is a portfolio-grade end-to-end build. The honest list of what
would change for real production deployment:

- **Streaming feature store.** Velocity features here are computed offline
  in pandas. In production these need a low-latency feature store (e.g.
  Redis-backed Feast, Tecton) keyed on `card1` / `card_hash`, populated by
  a streaming job (Flink/Kafka) so per-request lookup is O(1) and the
  feature value is the same offline (training) and online (serving) — the
  cardinal cause of "trained well, serves badly" model rot.
- **Model registry + CI/CD for models.** MLflow or W&B model registry,
  with a "challenger vs. champion" promotion gate that requires the new
  model to win on a recent shadow-traffic slice across PR-AUC *and* a
  cohort-stratified false-positive metric (don't ship a model that's better
  on average but worse on a specific BIN range).
- **A/B testing thresholds.** Threshold tuning is a per-merchant, per-BIN,
  per-channel knob, not a global one. Treat `policy.json` as an experiment
  config — ramp new policies on a percentage of traffic, log the
  counterfactual (what the old policy would have done), and compare on
  approval rate, chargeback rate, and review-queue volume.
- **Drift triggers.** The Evidently report here is run-on-demand. In prod,
  schedule it hourly/daily on a rolling window of production scoring logs
  joined to delayed labels (chargebacks land 30–90 days later, so target
  drift necessarily lags data drift). Alert on PSI > 0.2 on top features
  *or* a sustained PR-AUC drop on the labelled tail.
- **Adversarial dynamics.** Fraud is an adversarial process — fraudsters
  adapt. Retrain weekly, not quarterly. Track the model's calibration over
  time, not just discrimination — Platt scaling drifts faster than
  rank-ordering does.
- **Explainability for chargebacks.** Top-5 SHAP features in JSON is fine
  for analysts. For chargeback representment you'd render a templated
  explanation: "this order was flagged because the issuing-bank country
  differs from the shipping country **and** the device fingerprint had
  not been seen before for this account."
- **PII & regulatory.** The IEEE-CIS data is anonymised. A real system
  needs encryption-at-rest of card hashes, audit logging of every
  decision for a regulator-friendly retention window, and care with
  explanation responses — this service currently returns the raw `value`
  alongside each SHAP contribution, which echoes potentially sensitive
  feature values back to the caller. Production should redact `value` for
  external callers and surface it only in the analyst console.
- **Serving footprint.** The current Docker image bundles SHAP + XGBoost
  + Evidently. Production split: a light scoring service (XGBoost only,
  ONNX-converted for ~3× latency improvement) and an async explanation
  service that backfills SHAP values into the analyst review queue rather
  than blocking the synchronous decision.

---

## Tech stack

Python 3.11 · pandas · scikit-learn · XGBoost · SHAP · FastAPI · Pydantic v2 ·
Streamlit · Plotly · Evidently · joblib · Docker.
