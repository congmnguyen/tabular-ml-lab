# Experiment plan — EV purchases

Recorded on 2026-09-09. Model selection uses mean ROC-AUC on five fixed stratified folds (seed 42); the leaderboard is only an external check. These folds are reused for development, so the selected CV score is not an unbiased final estimate.

1. Logistic regression: linear/additive baseline with scaled numeric and one-hot categorical features.
2. LightGBM: test whether nonlinearities improve on the additive baseline.
3. CatBoost: compare native categorical handling with LightGBM's ordinal representation.
4. LightGBM regularized variant: reduce leaves from 31 to 15, increase minimum leaf observations and L2 penalty, lower learning rate and increase tree count. Hypothesis: smoother trees generalize better on this synthetic dataset.
5. LightGBM with stateless features: total nearby charging stations, commute per (cars + 1), log income, and home-charging × commute. Hypothesis: charging access and commute interactions add signal beyond raw columns.

No leaderboard-guided parameter search, external training data, pseudo-labels or stacking. Choose the best single run by mean fold AUC; retain all negative results. Train all candidates with fixed iteration budgets (no validation-driven early stopping). Evaluate log loss, average precision, calibration, and segment errors as secondary diagnostics, without choosing a deployment threshold.

Initial scope: one competition, binary classification, CPU models. Group/time split and regression support are not claimed.
