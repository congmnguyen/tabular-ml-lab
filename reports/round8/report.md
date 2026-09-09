# Round 8 — shared error regions

Starting public best: **0.94632**. Target: **0.94672**.

## Predeclared diagnostic

Fit a frozen version of round six's lower-smoothing modulo offset XGBoost on old folds 0–2 only, using a fixed 600-tree budget, with no early stopping on either diagnostic holdout. Reuse round five's frozen original-feature offset XGBoost, which used the same training rows. This lets us check whether error directions persist across two feature/encoding variants without overlapping diagnostic-label training. These are related XGBoost variants, not independent model families.

Discover broad error regions on fold 3 with one depth-three regression tree, minimum 3,000 discovery rows per leaf. Features are raw covariates, the mechanism score, income modulo 1000, training-only income frequency, and public-source income target means. Fit a Newton residual response with Bernoulli-variance weights. Each leaf correction is regularized by variance-sum +20, clipped to ±1 log-odds and applied with a fixed step of 0.5. Evaluate this single configuration on fold 4. Report both current and older frozen models' residual directions in each region.

The confirmation rows did not train the base models, discovery tree or leaf corrections. However, folds 3 and 4 have been inspected in earlier diagnostic rounds and all data influenced previous model choices: they are reused development holdouts, not untouched final tests. No significance or generalization guarantee follows from one positive check.

A confirmation AUC gain greater than **0.00002**, accompanied by interpretable shared residual directions, would justify a separately cross-fitted model experiment. Otherwise do not submit this correction or tune it against confirmation labels. Calibration improvement alone is insufficient for the AUC competition.

## Reproduction

```bash
python -m src.shared_error_audit
```

This requires the round-five frozen model's `heldout.csv` and round-four fold assignments. Source data and checkpoints remain local ignored artifacts.

## Findings

The tree identifies three regions whose discovery error direction repeats on confirmation for both frozen model variants:

| Region (discovered on fold 3) | Confirmation rows | Actual purchase rate | Predicted rate |
|---|---:|---:|---:|
| Mechanism score 1.16007–4.81510; at most 4 work chargers | 34,402 | 2.328% | 2.547% |
| Mechanism score above 4.81510; at most 4 work chargers | 15,357 | 54.106% | 54.328% |
| Mechanism score above 1.16007; at least 5 work chargers; income appears at most 16 times in base training | 4,707 | 22.583% | 23.049% |

These thresholds are exploratory tree splits, not physical discontinuities or causal effects. Only three of five leaf error directions replicate; a very-low-mechanism-score region reverses direction and has zero confirmation positives. Do not infer guaranteed negative labels from this sample. Tree text displays unregularized Newton leaf values; deployed diagnostic corrections use the separate ridge/clipping/step formula documented above.

The fixed correction changes confirmation AUC from **0.945904098** to **0.945915219**, a gain of **0.000011121**, below the predeclared **0.00002** expansion criterion. Log loss improves from **0.218707158** to **0.218676254**. This is a small calibration improvement with limited ranking benefit, not evidence of a large missing predictive mechanism.

## Decision

Do not expand this correction to competition folds or submit it. The selected model remains round seven, public AUC **0.94632**. This diagnostic compares related XGBoost variants trained on 60% of rows, so it does not establish the size of any remaining error in the selected 90%-training models or prove that all model families share these errors.

We have identified interpretable development error regions, but this experiment does not support a claim that correcting them will deliver a leaderboard jump. Further progress needs a distinct, testable hypothesis rather than fitting more thresholds to this confirmation set.

## Verification

Reloading the frozen model reproduces all **267,466** held-out predictions exactly after the same CSV serialization (maximum difference **0.0**). Direct comparison of raw float32 values with parsed decimal CSV values initially differed by up to 2.98e-8; matching serialization resolves that representation difference. Actual model and target-encoder parameters, artifact hash and script hash are saved in `provenance.json`. Existing tests: **10 passed**.
