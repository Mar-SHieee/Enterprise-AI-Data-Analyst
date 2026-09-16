# Member 3  owned by Mahmoud Walid— Classical ML Track

## Objective
Predict whether a customer will make a repeat purchase using behavior from the customer's first observed order.

## Data and target
The model uses the processed retail transaction schema produced by the Data/Spark tracks. One row is created per customer. The target `repeat_customer` is 1 when the customer has more than one observed invoice in the available data window.

To reduce target leakage, all predictive features are calculated from the customer's **first observed invoice only**:
- first-order revenue
- first-order quantity
- average item price
- unique products
- number of lines
- log-transformed versions of skewed numeric features

## Modeling
Two classical baselines are evaluated:
1. Logistic Regression with class balancing.
2. Random Forest with class balancing.

Three-fold stratified cross-validation is used on the training split. Because repeat customers are a minority class, Average Precision, ROC-AUC, F1, recall and balanced accuracy are tracked rather than relying on accuracy alone.

The Random Forest is tuned over tree depth, minimum leaf size and feature subsampling using Average Precision as the grid-search objective.

## Explainability
Permutation importance is calculated on the held-out test set. This shows which first-order customer attributes most affect model performance when shuffled.

## Business KPI simulation
The notebook includes a configurable retention-campaign scenario. It targets the highest predicted repeat probabilities and estimates incremental revenue from:
`customers targeted × assumed conversion uplift × assumed average repeat-order value`.

The result is explicitly a scenario estimate, not a causal impact measurement.

## Artifacts
The final estimator is saved as:
- `models/classical_repeat_purchase_rf_v1.joblib`
- `models/classical_repeat_purchase_rf_v1.json`

The JSON stores the task name, timestamp, feature schema and evaluation metrics for downstream MLOps/deployment.

## Limitations
The repository's available processed sample is only 300 transaction rows. Therefore, model metrics from this checkout are illustrative. For the full Online Retail II dataset, the notebook should be rerun and a chronological/time-based validation strategy should be considered.
