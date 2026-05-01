# Intelligent Order Risk & Routing System

A fraud detection and decision automation service for e-commerce orders. Each inbound order receives a calibrated risk score from an XGBoost model, gets routed to one of three outcomes (auto-approve, manual review, auto-reject), and arrives with a SHAP-based explanation suitable for analysts and chargeback evidence.

The model trains on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset of roughly 590,000 transactions with a 3.4% fraud rate.

## The business problem

Card-not-present fraud carries asymmetric costs.

| Outcome | What it costs |
|---|---|
| False negative (approving a fraud) | The full transaction value, a chargeback fee of $15 to $50, and a scheme penalty if the merchant fraud rate trips a threshold. A direct hit to P&L. |
| False positive (rejecting a good order) | The order revenue, plus the lifetime value of a customer who hits a "your card was declined" page. Industry benchmarks place the LTV cost at five to ten times the order value. |
| Manual review | Analyst time, roughly $2 to $5 per case. Cheap relative to either error, provided the queue stays sized for capacity. |

A single point estimate ("score above 0.5 blocks") collapses this nuance. Risk teams need four things instead.

1. A calibrated probability they can reason about. A score of 0.7 should mean that 70% of orders like this are fraudulent, which differs from "this came out higher than the other one."
2. Configurable thresholds, so operations can re-tune as the fraud mix shifts. The model artifacts stay frozen between tunings.
3. A review band, so the model hands low-confidence cases to humans for adjudication.
4. A reason for every decision, for analyst efficiency, chargeback representment, and regulatory defensibility.

This service provides all four.

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

Repository layout:

```
order-risk-engine/
├── data/                   # CSVs go here (gitignored)
├── pipeline/               # data loading, feature engineering, sklearn ColumnTransformer
├── model/
│   ├── train.py            # end-to-end training and calibration
│   ├── decision.py         # DecisionPolicy and threshold sweep
│   ├── explain.py          # SHAP TreeExplainer wrapper
│   └── artifacts/          # joblib and json outputs
├── api/                    # FastAPI app
├── dashboard/              # Streamlit operator console
├── monitoring/             # Evidently drift report
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Get the data

Download the IEEE-CIS competition data and unzip it into `data/`.

```bash
# requires `kaggle` CLI configured
kaggle competitions download -c ieee-fraud-detection -p data/
unzip -o data/ieee-fraud-detection.zip -d data/
```

You can also download the files manually from <https://www.kaggle.com/competitions/ieee-fraud-detection/data> and place `train_transaction.csv` and `train_identity.csv` in `data/`. The schema lives in `data/README.md`.

### 2. Train

```bash
python -m model.train
```

This step:

- merges the transaction and identity tables on `TransactionID`,
- engineers velocity features (per-card rolling count, sum, and mean over the last 5, 20, and 100 transactions),
- splits chronologically (60/20/20 train, calibration, validation), since random splits leak future information on this dataset,
- fits XGBoost with `scale_pos_weight = neg/pos`,
- wraps the trained pipeline with `CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")` for Platt scaling on the held-out calibration slice,
- writes the artifacts:
  - `model/artifacts/calibrated_pipeline.joblib`
  - `model/artifacts/booster.joblib` (raw booster for SHAP)
  - `model/artifacts/preprocessor.joblib`
  - `model/artifacts/feature_names.joblib`
  - `model/artifacts/metrics.json` (ROC-AUC, PR-AUC, F1, threshold sweep)
  - `model/artifacts/policy.json` (default thresholds)
  - `model/artifacts/validation_sample.parquet` (reference frame for the drift monitor)

Performance on the IEEE-CIS held-out chronological validation slice (118k rows, around 3.4% positives):

| Metric | Value |
|---|---|
| ROC-AUC | 0.874 |
| PR-AUC | 0.450 (roughly 13× the no-skill baseline of 0.034) |
| F1 at default `reject_above=0.65` | 0.420 |
| Brier score | 0.023 (against a naive baseline of 0.032, so calibration adds real lift) |

At the default reject threshold the model blocks roughly 1.3% of orders at 78% precision and 29% recall. The full precision, recall, and F1 sweep across thresholds 0.05 to 0.95 lives in `metrics.json` and renders in the dashboard's "Threshold tradeoff" tab.

Numbers vary slightly with random seed and XGBoost version.

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

Top SHAP contributors come overwhelmingly from Vesta's anonymized count and timedelta columns (`C*`, `D*`). The "Feature store dependency" section below explains why.

### 4. Operator dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard lets an operator submit a sample order, see the score and decision, visualize SHAP contributions, and re-tune thresholds against the saved precision and recall sweep. The model stays frozen throughout.

### 5. Drift monitoring

```bash
python -m monitoring.drift_report
open monitoring/reports/drift_report.html
```

This generates a synthetic-drift comparison (amount inflation, device-mix shift, card-testing-style velocity spikes) against the held-out validation slice using Evidently's data drift, target drift, and classification presets.

### 6. Tests

```bash
pytest -q
```

The included tests cover the decision policy and the feature-engineering invariants. They run on stub data alone, so the suite passes on a fresh clone before any training has happened.

## Decision policy

Default thresholds live in `model/artifacts/policy.json`.

| Score band     | Decision        |
|----------------|-----------------|
| `≥ 0.65`       | `auto_reject`   |
| `0.05 – 0.65`  | `manual_review` |
| `< 0.05`       | `auto_approve`  |

These defaults reflect the calibrated score distribution on this dataset. Platt scaling on rare-positive data (~3.4% fraud) compresses scores. The maximum score on the held-out validation slice sits at 0.76, so a 0.85 reject threshold would lie above the entire observed range. The thresholds above came from the actual validation sweep: `reject_above=0.65` blocks roughly 1.3% of orders at 78% precision and 29% recall, and orders below `approve_below=0.05` show an observed fraud rate of 1.25%, low enough to auto-approve with confidence.

`metrics.json` holds a precision, recall, and F1 sweep across `reject_above` from 0.05 to 0.95. The dashboard's "Threshold tradeoff" tab walks that curve interactively, so operations can pick a setting based on the current cost balance (chargeback rate, manual-review queue capacity, false-positive tolerance).

Thresholds live separately from the model, so risk operations can re-tune them while the model artifacts stay frozen.

## Feature store dependency

Most of the discriminative signal in the IEEE-CIS dataset comes from Vesta's anonymized count and timedelta columns (`C1..C14`, `D1..D15`) along with the engineered velocity features. SHAP confirms it: the top 5 contributing features for almost every prediction draw from `C13`, `C1`, `D1`, and the velocity counts.

This shapes the API contract.

- An API call that supplies only the human-meaningful fields (`TransactionAmt`, `ProductCD`, `card4`, `P_emaildomain`, `DeviceType`, and so on) will score nearly every order at the base rate. The model needs the recent-history signal to differentiate orders.
- A useful production caller supplies, or the API looks up, per-card recent-history features (counts on the card in the last N hours, time since last transaction, and so on) keyed on `card1` or a card hash.

The repo ships a sample payload (`scripts/sample_payload.py`) that includes representative `C*` and `D*` values, so the model can actually discriminate when you exercise the demo. In a real deployment those values arrive from a streaming feature store (Feast, Tecton, or a homegrown Redis plus Flink setup). The "Production scale" section below covers the rest of the production path.

## Production scale

The following changes carry this design from a working end-to-end build into a production deployment.

- Streaming feature store. Velocity features here are computed offline in pandas. Production needs a low-latency feature store such as Redis-backed Feast or Tecton, keyed on `card1` or a card hash, populated by a streaming job (Flink, Kafka) so per-request lookup runs in O(1) and the feature value matches between training and serving. Feature parity between offline and online is the cardinal cause of "trained well, serves badly" model rot.
- Model registry plus CI/CD for models. MLflow or Weights & Biases model registry, with a challenger-versus-champion promotion gate. The new model must win on a recent shadow-traffic slice across PR-AUC and a cohort-stratified false-positive metric, so a candidate that is better on average but worse on a specific BIN range fails the gate.
- A/B testing thresholds. Threshold tuning belongs at the per-merchant, per-BIN, per-channel level, finer-grained than a single global setting. Treat `policy.json` as an experiment config: ramp new policies on a percentage of traffic, log the counterfactual (the decision the prior policy would have made), and compare on approval rate, chargeback rate, and review-queue volume.
- Drift triggers. The Evidently report runs on demand here. In production, schedule it hourly or daily on a rolling window of production scoring logs joined to delayed labels (chargebacks land 30 to 90 days later, so target drift necessarily lags data drift). Alert on PSI above 0.2 on top features, or on a sustained PR-AUC drop on the labelled tail.
- Adversarial dynamics. Fraud is an adversarial process. Fraudsters adapt, so a weekly retraining cadence keeps pace with the threat surface where a quarterly cadence falls behind. Track calibration over time alongside discrimination, since Platt scaling drifts faster than rank-ordering.
- Explainability for chargebacks. The top-5 SHAP features in JSON serve analysts well. For chargeback representment, render a templated explanation, for example: "this order was flagged because the issuing-bank country differs from the shipping country, and the device fingerprint was new on this account."
- PII and regulatory. The IEEE-CIS data is anonymized. A real system needs encryption at rest for card hashes, audit logging of every decision for a regulator-friendly retention window, and care with explanation responses. This service currently returns the raw `value` alongside each SHAP contribution, which echoes potentially sensitive feature values back to the caller. Production redacts `value` for external callers and surfaces it only in the analyst console.
- Serving footprint. The current Docker image bundles SHAP, XGBoost, and Evidently. Production splits this in two: a light scoring service (XGBoost only, ONNX-converted for around 3× latency improvement) and an async explanation service that backfills SHAP values into the analyst review queue, so the synchronous decision path stays fast.

## Tech stack

Python 3.11, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic v2, Streamlit, Plotly, Evidently, joblib, Docker.
