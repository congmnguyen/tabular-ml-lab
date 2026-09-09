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

## Additional hypothesis after expert-feature screening

The full expert-inspired feature set improved fold-0 AUC to 0.945073 (LightGBM) / 0.945022 (XGBoost), versus 0.940630 for the previous selected model on this fold. Both proceed to five-fold evaluation.

Next screen: add auto-smoothed target encodings, original-to-competition frequency ratios and an inner-cross-fitted encoding of `target - original_recipe_probability` for income/commute keys. Rationale: isolate synthetic-generator deviations after accounting for the source recipe. Ratios learn competition frequencies from the outer train fold only. This grouped experiment cannot attribute its gain to one component; if useful, retain that limitation in the report.

A second focused screen uses 1,024 histogram bins (instead of 255), 0.3 feature fraction and minimum leaf size 10, following the published Pure LGBM settings. Hypothesis: preserve narrow repeated-income pockets while reducing correlated-feature competition at each tree. The exact applied settings are available from saved model get_params; the shared runner's initial params field is overridden by src/fine_bins.py for this candidate.

The residual/replication grouped screen scored 0.945032 on fold 0, below the 0.945073 full-feature LightGBM screen, so it was not expanded. A shallower XGBoost screen (depth 4, learning rate 0.025) tests smoother corrections and potential complementary errors. Actual estimator parameters, encoder classes and artifact hashes are exported with `python -m src.audit_expert_artifacts` to avoid relying on the shared runner's pre-override parameter dictionary.

Because explicit-TE LightGBM/XGBoost OOF probabilities correlate at 0.99932, their blend offers little diversity. A final screen tests GPU CatBoost with native categorical income/commute keys and original-data statistics, retaining digit/recipe features but replacing explicit cross-fitted target encoding with CatBoost's native categorical mechanism. Evaluate both its own fold-0 AUC and a fixed 25% CatBoost / 75% existing blend before expanding. GPU CatBoost is not bitwise deterministic.

After the 0.94618 public result, one representation ablation removes high-cardinality recipe/worry/interaction keys and the car-count family, while adding auto-smoothed target encoding to the raw/digit/bin keys and using the finer histogram setting. Motivation: target-encoding nearly unique derived scores may add noise and obscure repeated raw-value effects. This is a grouped, expert-informed ablation rather than a claim that each removed feature is useless. It is screened on fold 0 before any further submissions.

One important methodological difference from Naji's notebook is frequency estimation: its maps use combined train+test features, while ours originally used outer-training features only. A labelled-as-transductive ablation estimates frequencies from all available train/test feature rows (target excluded when reading CSV), while keeping every supervised target encoder cross-fitted within the outer training fold. This tests whether reduced frequency-estimation noise matters. Such a model assumes the competition test batch is available and is not an ordinary inductive deployment pipeline.
