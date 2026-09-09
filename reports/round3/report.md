# Round 3 — testing conditional effects and model diversity

Best public submission entering this round: **0.94618**, below the requested **0.94672** target. The round-two LightGBM/XGBoost probability blend scores 0.945090357 on development fold 0. All screens below use that same split; these are selection estimates, not an independent holdout.

## Research and hypotheses

Repeated exact income values contain signal, but a marginal target mean may confound income effects with subsidy, concern and anxiety. Two experiments test alternatives: conditional target encodings and jointly fitted sparse coefficients for exact numeric values, accompanied by smooth spline effects.

For a different model family, we inspected [yhay81's public TabM baseline](https://www.kaggle.com/code/yhay81/simple-tabm-baseline-s6e9) and [Tamerlan Omralinov's Offset Residual Net notebook](https://www.kaggle.com/code/tamerlanomralinov/s6e9-the-best-dl-model-insights). The former uses numeric digit views and an efficient small TabM ensemble. The latter reports that its neural variants trail boosting; its title is not evidence of leaderboard superiority. No notebook predictions were downloaded or used.

Our independent TabM screen uses the [official TabM package](https://github.com/yandex-research/tabm), compact raw/recipe/digit views, original-data means and inner-cross-fitted exact-income/commute target encodings. Quantile scaling is fitted only on each outer training fold. It retains the existing split and saves preprocessing, model weights and validation predictions. Binary cross entropy is applied separately to each ensemble member; inference averages probabilities. The purpose is to measure complementary errors, not assume that neural models win.

## Initial screens

| Model | Fold 0 AUC | Decision |
|---|---:|---|
| Existing two-tree probability blend | 0.945090357 | Reference |
| Conditional-key XGBoost | 0.944914899 | No expansion; 25% blend adds only 0.000010691 |
| Sparse exact-value logistic, C=0.1 | 0.941772156 | Reject; blending worsens reference |
| Sparse exact-value logistic, C=1 | 0.943124953 | 700-iteration limit reached; 5% blend adds only 0.000008290 |

Small gains on this repeatedly reused fold do not justify a leaderboard submission. Conditional keys create sparse groups and may increase estimation noise; this is an interpretation of the result, not a proven causal explanation. Further runs and their blend screens are recorded in `screens.json`.

The C=1 sparse model was rerun to convergence (929 iterations), reaching 0.943139913; its best screened blend improves the reference by only 0.000008850. C=10 reached 0.942672122 before the 700-iteration cap. Adding the public original rows to LightGBM with weight 0.25 reached 0.944746645 and did not improve the reference blend. These results do not warrant five-fold expansion.

Reproduction (from the repository root):

```bash
.venv/bin/python -m src.conditional --output artifacts/r3-conditional --model xgb --folds 0 --rounds 6000
.venv/bin/python -m src.sparse_values --output artifacts/r3-sparse-c1-converged --c 1 --iterations 3000
.venv/bin/python -m src.original_augmentation --output artifacts/r3-original-aug --model lgb --folds 0 --rounds 6000
.venv/bin/python -m src.screen_round3
```
