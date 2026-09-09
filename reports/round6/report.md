# Round 6 — smooth residuals and joint digit fingerprints

Starting public best: **0.94624**; target **0.94672**.

## Smooth income corrections

We used the frozen model from round five, fitted on folds 0–2. On its unseen folds 3 and 4, an eight-knot cubic spline basis represents income together with subsidy and concern interactions. A regularized one-step log-odds correction is fitted on one diagnostic fold and evaluated on the other. We check both directions, with penalties 10, 100 and 1000; neither evaluated fold's labels enter its correction fit.

The strongest regularization gives AUC gains of only **0.00000447** and **0.00000423**. Penalty 10 worsens AUC in both directions; penalty 100 gives +0.00000174 and +0.00000630. We did not expand these tiny, development-only gains into a competition submission. See `smooth-audit.json`.

## Public research and independent verification

We read these public notebooks for ideas; none was executed and no external prediction files were imported:

- [megayak: remainder-based pipeline](https://www.kaggle.com/code/megayak/the-1000-leak-a-from-scratch-0-946-pipeline): joint numeric remainders, multiresolution income bins and lower target-encoding smoothing.
- [megayak: six missing sources](https://www.kaggle.com/code/megayak/s6e9-lb-0-94643-and-six-missing-sources): its headline score relies on other authors' prediction files; it is not a from-scratch model result.
- [Dariush Afshar: phenomenon versus generator](https://www.kaggle.com/code/dariushafshar/s6e9-0-938-phenomenon-0-946-generator): source-generation verification and mechanistic baselines.
- [yhay81: exact source reconstruction](https://www.kaggle.com/code/yhay81/exact-reconstruction-of-the-ev-source-dataset): the original random-draw sequence.

We do not adopt the notebooks' broad claims of an established AUC ceiling, a universal noise floor, or impossibility of real-data digit associations. Those claims do not follow from our measurements.

Our own five-fold, target-encoded univariate check gives AUC **0.545854** for income modulo 10, **0.572532** for modulo 100, and **0.618058** for modulo 1000. This verifies predictive association, not causality or incremental benefit over the current model. The incumbent already has individual-digit features; the new hypothesis is that encoding three low-order digits jointly can capture additional value-level structure.

We also independently implemented the published source RNG sequence and verified every observed cell across twelve generated covariates, with zero mismatches. We filled only the source's missing income (178), commute (181) and concern (184) cells. All public source labels remain unchanged. This creates only **17 newly matched train incomes and 6 test incomes**, with no new commute matches, so its coverage benefit is small. Existing source group means can still change, motivating one direct screen rather than assuming an improvement. See `source-recovery.json`.

## Model screens

Both screens retain the incumbent's fold-trained probit offset and XGBoost settings:

1. Add income modulo 100 and 1000, commute-tenths modulo 100, and income bins of 250, 500 and 2500. The existing encoder supplies outer-train-only frequencies and inner-cross-fitted target means for the new keys.
2. Use the feature-recovered public original dataset for original-data means, leaving competition data unchanged.

Screens use the existing development fold zero. Only a promising candidate will be expanded to the remaining folds. Repeated model selection and early stopping remain sources of optimism; public scores are measured separately.

Initial screens: joint remainder/multiresolution features reached fold-zero AUC **0.945116826**, slightly below the incumbent offset's **0.945129992**. Recovered original features reached **0.945150230**, a small +0.000020238 screen gain; this candidate is checked on all remaining folds. A third screen retains the joint remainder features but lowers target-encoder smoothing from **10/100 to 2/20**, testing whether shrinkage masks sparse repeated-value signal. This follows the public notebook's hypothesis, not its claimed scores.

The lower-smoothing screen reached **0.945159662**, a +0.000029670 gain over the incumbent offset on fold zero, and was expanded. We also test a fourth hypothesis: target-statistic precision. With five outer and five inner folds, each training-row target mean sees about **64% of all competition training rows**; twenty inner folds raises this to about **76%**, while outer validation labels remain excluded. The twenty-inner-fold screen retains the original features and smoothing 10/100, isolating this precision change from the modulo and smoothing experiments. This can reduce train/inference encoding mismatch, but improvement must be measured.

## Completed validation

| Candidate | Pooled five-fold OOF AUC | Decision |
|---|---:|---|
| Previous offset XGBoost | 0.946076181 | Reference |
| Recovered source features | 0.946077459 | Essentially tied; not selected |
| Modulo features with smoothing 2/20 | 0.946130159 | Selected for blending |

Twenty inner folds reached only **0.945076618** on fold zero, below the incumbent **0.945129992**, and were not expanded. More observations per target statistic did not help this screen.

The existing eight-candidate blend grid selected rank weights **45% LightGBM / 45% lower-smoothing offset XGBoost / 10% TabM**. Mean fold AUC is **0.946166351**, pooled OOF AUC **0.946166530**. Mean fold AUC improves by **0.000040319** over the round-four blend, with positive differences in all five folds. This is development evidence, not an independent estimate after repeated selection. The winning XGBoost changes both features and smoothing; we have not isolated their individual contributions.

Actual saved target-encoder settings are exported in `runs/`, including smoothing and inner fold counts. The source-recovery candidate is separate from the selected model, which uses the original public source file.

## Reproduction

```bash
python -m src.low_smoothing --model xgb --output artifacts/r6-low-smoothing
python -m src.blend_experts --runs artifacts/r2-full-lgb artifacts/r6-low-smoothing artifacts/r3-tabm --small-third --output artifacts/r6-blend
python -m src.predict_expert --run artifacts/r6-blend --test /home/cong/.cache/kaggle/playground-series-s6e9/test.csv --output artifacts/r6-blend/reproduced.csv
python -m src.audit_expert_artifacts --prefix r6 --report reports/round6/runs
python -m pytest -q
```

Install the base and optional neural requirements before reproducing all blend components. Earlier rounds document their training commands. Model checkpoints and competition data remain local ignored artifacts.

Saved-model inference reproduced all **286,571** submission rows with maximum absolute difference **0.0**. Existing tests: **8 passed**. Kaggle submission ID: **56121758**.

Public AUC: **0.94627**, a **+0.00003** improvement over round four. The requested **0.94672** target remains unmet, with a gap of **0.00045**. Public leaderboard feedback is also development feedback; no private-label performance claim is made.

The next useful hypothesis is greater outer-fold training coverage (for example ten folds, each training on 90% of rows), which differs from the unsuccessful increase in inner target-encoding folds. It requires its own honest OOF partition and inference aggregation; it has not been tested in this round.
