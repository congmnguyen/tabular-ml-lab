# Round 9 — jointly adjusted value effects

Starting best public AUC: 0.94632. Target: 0.94672.

Round four estimated income and commute residual effects separately around a probit mechanism; it failed the fold-zero screen. Independent estimates can each absorb variation attributable to the other variable. This experiment estimates both sets of additive log-odds effects jointly, using alternating penalized Newton updates, while retaining the fold-fitted probit starting risk. Penalty is fixed at 2, maximum 30 passes, per-update clipping ±0.5, tolerance 1e-5.

Two effects are appended to the current lower-smoothing modulo feature set. Training effects use five inner stratified folds; outer validation and test use mappings fitted only on outer training data. Unknown values receive zero effect. The existing probit-offset XGBoost parameters remain unchanged.

Screen only ten-fold partition zero against round seven's AUC **0.945084147**. Expand to all ten folds only for a screen gain of at least **0.00002**. This is a development screen, not independent confirmation. The experiment changes both joint fitting and regularization compared with the older failed feature, so their separate contributions cannot be isolated from this comparison.

```bash
python -m src.joint_effects --model xgb --splits 10 --folds 0 --output artifacts/r9-joint
```

The first screen reached **0.945133753**, a gain of **0.000049605**, meeting the expansion criterion. The unchanged configuration was expanded to all ten folds. A controlled synthetic test confirms joint fitting can avoid assigning a correlated proxy the effect of another variable; this is an implementation check, not evidence of competition improvement.

Before full results, we also declare one complementarity check: compare the incumbent and candidate singles, probability weights 25/50/75%, and a 50:50 rank blend using the existing two-run grid. Both runs use the same ten outer partitions, allowing valid aligned OOF blending. The blend scorer now evaluates every observed fold instead of hard-coding five. Only submit a candidate with mean fold AUC gain greater than **0.00002** over the incumbent and positive pooled OOF gain. This guards against spending a submission on a near-tie; it is a practical threshold, not a statistical significance test.

## Full results

| Run | Pooled OOF AUC | Mean ten-fold AUC |
|---|---:|---:|
| Round-seven incumbent | 0.946221346 | 0.946224618 |
| Joint-effect features | 0.946223719 | 0.946226963 |
| Selected 50:50 rank blend | 0.946273122 | 0.946273342 |

The standalone candidate essentially ties the incumbent: pooled gain **0.000002373**. The fixed blend grid selects equal rank weights, with mean-fold gain **0.000048724** and pooled gain **0.000051776**, exceeding the predeclared submission threshold. Nine of ten fold AUCs improve. Predictions remain strongly correlated (**0.999230**), so this is modest complementarity, not a newly discovered independent source of signal.

On screen fold zero, the joint income effect has the highest mean split gain. That does not establish independent predictive information or causality; correlated features can substitute for one another. The full OOF results are more informative than that importance ranking.

```bash
python -m src.joint_effects --model xgb --splits 10 --output artifacts/r9-joint
python -m src.blend_experts --runs artifacts/r7-tenfold artifacts/r9-joint --output artifacts/r9-blend
python -m src.predict_expert --run artifacts/r9-blend --test /home/cong/.cache/kaggle/playground-series-s6e9/test.csv --output artifacts/r9-blend/reproduced.csv
python -m src.audit_expert_artifacts --prefix r9 --report reports/round9/runs
python -m pytest -q
```

## Public result and decision

Submission **56122886** scored **0.94631**, versus incumbent **0.94632**. The OOF-selected blend did not improve the public best; keep round seven as the best public result. Do not tune more weights against this single public score.

This does not establish that the new features always harm generalization: the displayed difference is only 0.00001, and both CV and public scores have selection and sampling limitations. It does establish that this submission did not beat the measured incumbent. The target **0.94672** remains unmet, gap **0.00040** from the best score.

Saved-model inference reproduced all **286,571** submission rows exactly, maximum difference **0.0**. Tests: **12 passed**. `selected.json` records the OOF-selected round-nine candidate, not a replacement for the best public model.
