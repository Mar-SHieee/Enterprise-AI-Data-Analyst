# Member 4 —Mahmoud Elaraby (Deep Learning Track)

## Objective
Build a deep-learning upgrade for the same repeat-purchase prediction problem
Member 3 solved with classical ML, and compare the two directly on identical
data, features, target and train/test split.

## Data and target
Reuses `build_repeat_purchase_dataset()` and `DEFAULT_FEATURES` from
`src/features/engineering.py` **without modification**. One row per customer;
the target `repeat_customer` is 1 when the customer has more than one
observed invoice in the available data window. All predictors come from the
customer's first observed invoice only (leakage-aware, same as the classical
track).

The train/test split is identical to the classical notebook:
`test_size=0.25, stratify=y, random_state=42`. A validation split is carved
out of the training portion only (`test_size=0.2`, same random state) purely
for early stopping — the held-out test set is never touched during training.

## Model
`ChurnMLP` (`src/models/train.py`): a 2-hidden-layer MLP (32 → 16 units) with
BatchNorm and Dropout(0.3), trained with:
- Class-weighted `BCEWithLogitsLoss` (weight = negative/positive ratio in the
  training split, since repeat customers are a minority class).
- Adam optimizer, `ReduceLROnPlateau` learning-rate scheduling.
- Early stopping on validation loss (patience = 20 epochs).

Features are standardized with `StandardScaler` fit on the training split
only.

## Results

**On `data/sample.csv` (261 customers, 66 in the test split) — same data
Member 3's saved model was evaluated on:**

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|
| Random Forest (Member 3, saved artifact) | 0.551 | 0.160 | 0.182 | 0.167 | 0.200 | 0.139 |
| MLP (Member 4) | 0.460 | 0.091 | 0.000 | 0.000 | 0.000 | 0.217 |

Both models are close to random here. This is a **data-volume problem, not a
modeling problem**: the test split has only ~6 positive (repeat) customers
out of 66, which is not enough signal for either a Random Forest or an MLP
to learn from reliably.

## Explainability
A dependency-free permutation-importance routine (shuffle one feature at a
time, measure the drop in PR-AUC) is used instead of SHAP, so the DL and
classical reports read the same way and no extra heavy dependency is
required. On the sample data, `log1p_first_order_quantity` and
`log1p_first_order_avg_price` show the largest (small) importance — consistent
with the limited signal available at this sample size.

## Artifacts
Saved with the same convention as Member 3's classical artifact, for Member
6 to load in `src/models/predict.py`:
- `models/dl_mlp_repeat_purchase_v1.pt` — model weights.
- `models/dl_mlp_repeat_purchase_v1.json` — feature schema, scaler
  parameters, architecture, and evaluation metrics.

## Limitations
- `data/sample.csv` has only 300 transaction rows / 261 customers, the same
  limitation Member 3 already documented. Metrics computed on it are
  illustrative, not representative.
- The full-dataset re-run (section 5b of the notebook) is a single train/test
  split, not cross-validated; a production version should add stratified CV
  and hyperparameter tuning for the MLP, mirroring what Member 3 did for the
  Random Forest.
- Calibration (Brier score) is reported but no explicit calibration step
  (e.g. Platt scaling / temperature scaling) has been applied yet.
