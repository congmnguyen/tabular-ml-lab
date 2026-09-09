# Round 7 — more outer-fold training data

Starting best public AUC: **0.94627**. Requested target: **0.94672**.

## Hypothesis and protocol

Keep the round-six offset XGBoost, joint remainder features, target-encoding smoothing 2/20, five inner folds and other hyperparameters. Change outer StratifiedKFold from five to ten (shuffle, seed 42). Each model now trains on 90% rather than 80% of competition rows; its inner cross-fitted statistics use approximately 72% rather than 64% of all rows. More repeated-income examples may improve sparse value estimates.

All ten held-out partitions are disjoint and cover every training row exactly once. Each prediction excludes that row from both model fitting and target-statistic fitting. Early stopping still uses the scored outer fold, so this remains development OOF, with selection optimism. Changing fold count changes both training size and partition/model-averaging effects; the result cannot isolate training size causally.

Compare pooled OOF AUC with the five-fold round-six single model (0.946130159) and selected ensemble (0.946166530). Do not compare raw mean fold AUC across different partitions as though the folds were paired. No mixed five/ten-fold OOF blend is used in this experiment. Submit the standalone ten-fold average only if its pooled OOF improves over the incumbent ensemble. This decision is set before observing the completed run.

## Reproduction

```bash
python -m src.low_smoothing --model xgb --splits 10 --output artifacts/r7-tenfold
python -m src.predict_expert --run artifacts/r7-tenfold --test /home/cong/.cache/kaggle/playground-series-s6e9/test.csv --output artifacts/r7-tenfold/reproduced.csv
python -m src.audit_expert_artifacts --prefix r7 --report reports/round7/runs
python -m pytest -q
```

The trainer retains five folds by default. Inference reads the saved split count, with five as the default for historical summaries. Regression tests cover both ten-fold inference and exact historical float32-divide/float64-accumulate behavior.

## Results

Ten-fold pooled OOF AUC: **0.946221346**. The same model under five folds scored **0.946130159**: gain **0.000091187**. The incumbent round-six ensemble scored approximately **0.946166530**: gain **0.000054817**. Minor sub-nanounit differences from earlier reports can arise from CSV parsing and rank ties.

When grouped by the old five-fold row assignments, the candidate improves over the same single model in all five groups and over the incumbent ensemble in four of five groups. These are descriptive same-row comparisons, not independent replicates or confidence intervals. See `comparison.json`. The predeclared standalone-submission criterion is met.

This supports further investigation of training coverage, but the observed improvement is much smaller than the approximately 0.00048 gain reported by the public reference notebook for its different pipeline. We do not extrapolate that external result to ours.

## Submission and verification

Saved-model inference reproduces all **286,571** submission rows with maximum absolute difference **0.0**. Tests: **10 passed**. Kaggle submission **56122041** scored **0.94632**, a **+0.00005** improvement over the previous public best **0.94627**. The requested **0.94672** remains unmet; gap **0.00040**.

The selected submission is the arithmetic mean of ten offset XGBoost fold models. It does not require LightGBM or TabM at inference. Public and development OOF improvements are consistent in direction, but neither provides an untouched final-test estimate after repeated experimentation.
