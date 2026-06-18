# KKBox Cost-Sensitive Churn Prediction

## Project Overview

This project predicts subscription churn for KKBox users and evaluates the model as a retention decision system, not as a raw accuracy exercise.

The business goal is to help a retention marketing team decide which customers should receive a targeted retention action before they fail to renew.

Churn is defined as:

> A user does not renew within 30 days after membership expiration.

## Business Framing

The model is designed for a retention marketing team that wants to prioritize high-risk subscribers for outreach.

Decision:

- score users using data available up to the prediction cutoff
- target users whose churn probability is above a selected threshold
- choose the threshold using business cost assumptions

Cost assumption for version 1:

- false positive cost = 1 unit
- false negative cost = 5 units

False positive:

- sending an offer to a user who would have renewed anyway

False negative:

- missing a user who actually churns

## Dataset

Source: KKBox Churn Prediction Challenge on Kaggle.

Version 1 uses:

- `train_v2.csv`
- `transactions.csv`
- `members_v3.csv`
- `WSDMChurnLabeller.scala` for understanding the churn label logic

Version 1 excludes:

- `transactions_v2.csv`
- `user_logs.csv`
- `user_logs_v2.csv`
- Kaggle submission files

Reason:

`transactions.csv` contains transactions up to 2017-02-28, matching the version 1 prediction cutoff. Later files may contain future information relative to the modeled decision point.

## Project Structure

```text
kkbox-churn/
|-- data/
|   |-- raw/
|   `-- processed/
|-- examples/
|   |-- create_sample_request.py
|   `-- sample_request.json
|-- models/
|   `-- hgb_churn_model_v1.joblib
|-- notebooks/
|-- reports/
|-- src/
|   |-- api/
|   |   |-- main.py
|   |   `-- schemas.py
|   |-- features/
|   |   `-- build_features.py
|   `-- models/
|       |-- train_model.py
|       `-- predict.py
|-- test_api/
|   `-- test_api.py
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- README.md
|-- requirements-api.txt
`-- requirements.txt
```

Raw data and processed CSVs are excluded from Git. The selected 579 KB model artifact is included because the API requires it at runtime.

## Main Workflow

Build features:

```bash
python src/features/build_features.py
```

Train selected model:

```bash
python src/models/train_model.py
```

Generate predictions:

```bash
python src/models/predict.py
```

## FastAPI Model Service

Version 1 includes a FastAPI service around the saved HGB model.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Confirm the API and model are available |
| POST | `/predict` | Score one customer |
| POST | `/predict-batch` | Score 1 to 1,000 customers |

The API returns churn probability, selected threshold, predicted class, risk band, and suggested retention action.

Important limitation:

The API expects the 30 engineered model features. It does not build features directly from raw transaction history.

### Run Locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the development server:

```powershell
python -m uvicorn src.api.main:app --reload
```

Open:

- API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Example Prediction

```powershell
$body = Get-Content examples\sample_request.json -Raw

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Example response:

```json
{
  "churn_probability": 0.4943,
  "prediction_threshold": 0.13,
  "predicted_churn": 1,
  "risk_band": "30-60%",
  "suggested_action": "moderate_retention_offer"
}
```

## Automated API Tests

Run:

```powershell
python -m pytest -v
```

The tests cover health, valid prediction, invalid input, batch prediction, and empty-batch rejection.

Current result:

```text
5 passed
```

## Docker

Build the API image:

```powershell
docker build -t kkbox-churn-api:v1 .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 --name kkbox-churn-api kkbox-churn-api:v1
```

Verify it from another terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

The Docker container was verified successfully using both `/health` and `/predict`.

## Feature Engineering

The final version 1 modeling table has:

- 970,960 users
- 32 columns
- 8.99% churn rate

Feature groups:

- member profile features
- cleaned age and missing-age flag
- registration tenure
- transaction count features
- cancellation and auto-renew counts
- payment summary features
- latest transaction features
- transaction-data availability flag

Important leakage rule:

Features must be available on or before 2017-02-28.

## Models Compared

Models evaluated on validation:

- Logistic Regression
- Logistic Regression with `class_weight="balanced"`
- HistGradientBoostingClassifier
- tuned HistGradientBoostingClassifier variants

The selected model is:

```text
HistGradientBoostingClassifier
max_iter = 150
learning_rate = 0.05
max_leaf_nodes = 31
l2_regularization = 0.1
threshold = 0.13
```

## Final Test Results

Selected model test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.9021 |
| Precision | 0.4732 |
| Recall | 0.7808 |
| F1 | 0.5893 |
| ROC-AUC | 0.9158 |
| PR-AUC | 0.7113 |
| Brier score | 0.0434 |

Test confusion matrix:

|  | Predicted Non-Churn | Predicted Churn |
|---|---:|---:|
| Actual Non-Churn | 161,545 | 15,181 |
| Actual Churn | 3,828 | 13,638 |

## Business Cost Comparison

Cost assumption:

- false positive cost = 1
- false negative cost = 5

| Strategy | FP | FN | TP | Cost Per Customer |
|---|---:|---:|---:|---:|
| Contact nobody | 0 | 17,466 | 0 | 0.4497 |
| Contact everybody | 176,726 | 0 | 17,466 | 0.9101 |
| Selected HGB threshold 0.13 | 15,181 | 3,828 | 13,638 | 0.1767 |

The selected model reduces expected cost per customer by about 60.7% compared with contacting nobody under the chosen cost assumption.

## Error Analysis Findings

Key findings:

- users with low transaction history have much higher churn
- users with no auto-renew history are high risk
- the model over-targets some manual-renew users
- false negatives often look historically stable, especially with strong auto-renew history
- risk bands are useful for assigning different retention actions

Risk band insight:

| Predicted Risk Band | Users | Actual Churn Rate | Action |
|---|---:|---:|---|
| 0-5% | 150,940 | 1.78% | no discount |
| 5-13% | 14,433 | 7.87% | no discount under current threshold |
| 13-30% | 12,263 | 21.21% | low-cost reminder |
| 30-60% | 7,473 | 42.33% | moderate retention offer |
| 60-100% | 9,083 | 86.69% | strongest retention action |

## Feature Importance

Top permutation importance features:

- `latest_transaction_days_before_cutoff`
- `latest_is_cancel`
- `latest_is_auto_renew`
- `latest_membership_expire_days_after_cutoff`
- `latest_payment_method_id`

Interpretation:

The model relies heavily on recent transaction timing and subscription behavior.

## Limitations

Version 1 limitations:

- split is stratified random, not true out-of-time validation
- user listening logs are excluded
- cost values are normalized assumptions, not confirmed business finance numbers
- encoded categories such as city and registration channel are not decoded
- extreme expiration-date values exist and should be cleaned more carefully in a future version
- the API accepts engineered features rather than raw customer transaction history
- Docker is verified locally but is not yet hosted on a public cloud service

## Next Improvements

Potential version 2 improvements:

- add listening behavior from `user_logs`
- clean extreme expiration-date features more carefully
- compare performance with and without dominant timing features
- add feature importance plots
- deploy the Docker container to a public cloud service
- add a small Streamlit interface that calls the API
- add authentication, logging, and monitoring for a production-style version
