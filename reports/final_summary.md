# KKBox Churn Prediction Final Summary

## Business Problem

KKBox wants to identify subscribers who are likely to churn so the retention marketing team can decide who should receive a targeted retention action.

In this project, churn means:

> The subscriber does not renew within 30 days after membership expiration.

The goal is not only to predict churn, but to support a better business decision than simple rules such as contacting nobody or contacting everybody.

## Decision Frame

The model scores users and assigns a churn probability. A customer is selected for retention action if their predicted churn probability is at least the chosen threshold.

Version 1 cost assumption:

- False positive cost = 1 unit
- False negative cost = 5 units

This means missing a true churner is assumed to be five times more expensive than sending an unnecessary offer.

## Data Used

Version 1 uses:

- `train_v2.csv`
- `transactions.csv`
- `members_v3.csv`

Version 1 excludes:

- user listening logs
- `transactions_v2.csv`
- Kaggle submission files

This keeps the first version focused on member and transaction behavior while avoiding future information leakage.

## Selected Model

Selected model:

```text
HistGradientBoostingClassifier
```

Selected threshold:

```text
0.13
```

The threshold was selected on the validation set using cost-sensitive evaluation.

## Final Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.9021 |
| Precision | 0.4732 |
| Recall | 0.7808 |
| F1 | 0.5893 |
| ROC-AUC | 0.9158 |
| PR-AUC | 0.7113 |
| Brier score | 0.0434 |

Confusion matrix:

|  | Predicted Non-Churn | Predicted Churn |
|---|---:|---:|
| Actual Non-Churn | 161,545 | 15,181 |
| Actual Churn | 3,828 | 13,638 |

Interpretation:

- The model catches about 78% of churners.
- About 47% of targeted users actually churn.
- The model misses 3,828 churners in the test set.
- The model sends unnecessary offers to 15,181 non-churners.

## Business Cost Comparison

Cost assumption:

- False positive cost = 1
- False negative cost = 5

| Strategy | FP | FN | TP | Cost Per Customer |
|---|---:|---:|---:|---:|
| Contact nobody | 0 | 17,466 | 0 | 0.4497 |
| Contact everybody | 176,726 | 0 | 17,466 | 0.9101 |
| Selected model, threshold 0.13 | 15,181 | 3,828 | 13,638 | 0.1767 |

The selected model reduces expected cost per customer by about 60.7% compared with contacting nobody under the chosen cost assumption.

## Risk Bands

Instead of treating every targeted user the same, customers can be grouped by predicted churn risk.

| Predicted Risk Band | Users | Actual Churn Rate | Suggested Action |
|---|---:|---:|---|
| 0-5% | 150,940 | 1.78% | No discount |
| 5-13% | 14,433 | 7.87% | No discount under current threshold |
| 13-30% | 12,263 | 21.21% | Low-cost reminder |
| 30-60% | 7,473 | 42.33% | Moderate retention offer |
| 60-100% | 9,083 | 86.69% | Strongest retention action |

## Main Findings

1. Transaction recency is the strongest signal.

Users whose latest transaction happened further before the cutoff are more likely to churn.

2. Auto-renew behavior is highly important.

Users with no auto-renew history are much riskier, but the model also over-targets some manual-renew users.

3. Cancellation behavior matters.

Latest cancellation and cancellation count are strong warning signals, but cancellation is not identical to churn.

4. Low transaction history means higher risk.

Users with few or no prior transactions have much higher churn rates.

5. Some churners look stable.

False negatives often have strong auto-renew history and no latest cancellation, making them difficult to detect using transaction features alone.

## Business Recommendations

Use the model as a prioritization tool, not as a simple yes/no machine.

Recommended approach:

- target users above the 0.13 threshold under the current cost assumption
- use stronger offers only for the highest risk band
- use low-cost reminders for medium-risk users
- avoid giving expensive discounts to all manual-renew users
- investigate auto-renew churners further because many missed churners look historically stable

## Limitations

Version 1 limitations:

- cost assumptions are normalized, not confirmed finance values
- split is stratified random, not true out-of-time validation
- user listening behavior is not included
- city and registration channel codes are not decoded
- extreme expiration-date values exist and should be cleaned more carefully in a future version
- the model relies heavily on timing features, which should be monitored in future versions

## Next Steps

Recommended version 2 improvements:

- add user listening behavior from `user_logs`
- clean extreme expiration-date values more carefully
- compare model performance with and without dominant timing features
- add visual charts for risk bands and feature importance
- create a Streamlit or FastAPI scoring demo
