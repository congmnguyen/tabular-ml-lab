# Round 2 — learn from experts, then test the data

The user requested an attempt to exceed the observed leader score of 0.94672 (Chris Deotte, checked 2026-09-09). This is a target, not a promised result. The initial five-model experiment is preserved unchanged; this round explicitly broadens its scope.

## Public sources read

- Chris Deotte: [original-data EDA](https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights), [XGB starter](https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter), and [Simpson's paradox](https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738991). Learn the generating mechanism, conditional relationships, and recipe-as-feature/base-margin comparisons. These public starters do not establish how the private leading submission was built.
- Naji: [Pure LGBM](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637). Ideas: decimal digits, original-data target means, multi-resolution bins, cross-fitted target encodings with several smoothing levels. The notebook credits cstdy, Evgeniy Dvorkin, Markus.JM, starkhushi/Tilii, broccoli beef and Chris Deotte.
- Mikhail Naumov: [Single XGB](https://www.kaggle.com/code/mikhailnaumov/electric-vehicle-purchases-single-xgb). Ideas: native categorical XGB, income bins, original statistics, frequency features and strongly regularized shallow boosting.
- Amirhossein Karimiee: [digit-feature ablation](https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739596). Synthetic artifacts may matter more than generic domain interactions.
- Original data: [EV Adoption Behavior and Range Anxiety](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety), 10,000 rows, CC0-1.0 as reported by Kaggle. Competition rules permit equally accessible external data. No external prediction files or private solutions are used.

Code is independently implemented from these documented ideas. Downloaded reference notebooks remain outside the repository and are not executed wholesale.

## Independent findings

Reproduce with `python -m src.investigate`. Full aggregates: [data-investigation.json](data-investigation.json).

- Recipe ranking AUC: 0.908477 on original complete income/concern cases; 0.937690 on competition train. The original recipe is useful, but its probabilities need correction on the resynthesized population.
- 668,665 training rows contain only 13,214 unique income values. Exactly 30,000 occurs 61,605 times (9.21%), versus 557/10,000 (5.57%) in the source data.
- The charging Simpson pattern reproduces in competition train. It does not reproduce identically in every original-data stratum, so we do not equate the two populations.
- Baseline OOF residuals vary with the last income digit (roughly -0.00446 to +0.00470 by group). This motivates an ablation; aggregate differences alone do not prove improvement.
- 393 train rows above the published 170,537 income boundary are positive. The published 38k–42k 'dead zone' is NOT exactly deterministic: its observed positive rate is 0.001566. We do not force test labels at these boundaries.
- No exact full-feature original-row matches were found in train or test. Original income, commute and concern contain missing values; competition data does not.

## Next experiments and validation

1. Expert recipe feature and worry score; test whether providing an appropriate nonlinear direction beats generic interaction features.
2. Add decimal digits, train-fold frequency encodings, multi-resolution income/commute keys, original-data statistics, and inner-cross-fitted target encoding. Test original-data augmentation separately if promising.
3. Compare shallow regularized LightGBM and GPU XGBoost. Screen on fold 0, then train promising configurations across all five fixed folds. Screening and early stopping make the resulting CV a development estimate, not an unbiased holdout.
4. If complementary models emerge, compare a small set of fixed blends on OOF, including per-fold deltas; submit only validated improvements. Maximum ten submissions/day; eight remained at the beginning of this round.

Target encoders fit_transform only on each outer training fold (internal five-fold cross-fitting); transform on outer validation/test. Original target statistics use only the public original dataset. Frequency maps and categorical dictionaries learn from outer training features, not validation labels. No leaderboard probing to recover labels, no private-data access, no public prediction-file averaging.
