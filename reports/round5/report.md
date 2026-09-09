# Round 5 — do income interactions replicate?

**Decision: do not expand income × subsidy/concern encodings on the current evidence.** No new competition submission was made. The best public score remains **0.94624** from round four.

The proposed next step was conditional: identify reproducible residual patterns first, then build strongly regularized group effects only if supported. This round completed that diagnostic and did not find sufficiently stable patterns among the selected large errors.

## Diagnostic design

We first inspected the offset model's existing OOF residuals. These are useful for exploration, but models across folds share training rows: labels held out from one model enter other models' training sets. Correlating their residuals is therefore not an independent replication test, and fitting global corrections to those residuals would risk leakage.

We trained a separate offset XGBoost once on **folds 0, 1 and 2**, with a fixed **600-tree** budget. It received no labels from folds 3 or 4 during fitting or early stopping; there was no early stopping in this fit. **Fold 3** supplies discovery residuals and **fold 4** supplies confirmation residuals, with **133,733 rows each**. Model settings were informed by earlier development experiments, so these are diagnostic holdouts, not a newly pristine final test.

For each side separately, the audit subtracts the estimated global subgroup residual effect, then estimates the remaining within-income deviation. Residual sums are scaled using the predicted Bernoulli variance; subgroup effects are shrunk with a denominator penalty of 20. These are approximate diagnostic scores, not causal effects or significance tests.

Eligibility requires **at least 30 rows and a predicted variance sum of at least 3 on each side**. Among eligible groups, we select at most 30 by the absolute standardized discovery error. Selection does not use confirmation residuals. Minimum confirmation support uses counts and model predictions, not confirmation labels.

## Frozen-model results

| Diagnostic | Income × subsidy | Income × concern |
|---|---:|---:|
| Eligible groups | 222 | 21 |
| Discovery rows covered | 23,534 / 133,733 | 10,170 / 133,733 |
| Spearman correlation of effects across sides | 0.1152 | 0.0727 |
| Same-sign effects across all eligible groups | 71.2% | 47.6% |
| Selected discovery groups | 30 | 21 |
| Selected groups retaining their sign in confirmation | **3 / 30** | **10 / 21** |

Overall sign agreement is higher for subsidy, but sign agreement alone does not establish a useful correction: effect ordering is weak and the selected largest discovery errors do not repeat reliably. Concern has too few eligible groups for a broad conclusion. Most discovery rows are outside the support threshold: about **82.4%** for subsidy and **92.4%** for concern. We should not generalize the eligible-group findings to those sparse cells.

This does **not** establish that conditional income effects are absent. It establishes that this diagnostic did not justify adding the proposed features or spending submissions on them. A frozen model trained on 60% of rows also differs from the incumbent's 80%-per-fold fits; that limitation remains.

Full aggregate results and selected groups are in `frozen-audit.json`. The comparable, non-independent OOF exploration is preserved separately in `oof-audit.json`. The frozen model's pooled diagnostic AUC is 0.945886326; it was never used as a competition submission or compared as a candidate on an equivalent five-fold score.

## Reproduction

```bash
.venv/bin/python -m src.frozen_diagnostic
.venv/bin/python -m src.audit_interactions --source oof
.venv/bin/python -m src.audit_interactions --source frozen
```

The model and row-level held-out predictions stay in ignored `artifacts/r5-frozen-diagnostic/`. The incumbent model and submission are unchanged. Avoid applying any residual table from this report directly to competition test predictions; these are diagnostic tables, not leakage-safe trained encoders.
