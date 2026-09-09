# Round 10 — source-row structure beyond income reuse

Starting public best: **0.94632**. Target: **0.94672**.

## Label-free protocol

Restrict to incomes that identify exactly one row in the public original source. Exclude ambiguous incomes (including the 30,000 floor) rather than choosing a source row arbitrarily. For train and test separately, measure exact agreement across all other covariates with that source row. Competition labels are not loaded.

Compare against ten fixed-seed controls that permute entire source covariate blocks within income bins of width 5,000. A second control also conditions on source city, preserving the city/charging associations that could otherwise inflate apparent source retention. Permuting whole blocks preserves source-internal covariate relationships. Report coverage, individual-column matches, total agreement, and high-agreement tails.

Source uniqueness is a convenience, not evidence that an original row generated a competition row. Positive agreement beyond controls supports testing source-context features; it does not establish record ancestry or justify inferring competition labels directly. A credible signal must replicate on unlabeled test covariates. Neither test labels nor external predictions are used.

```bash
python -m src.source_structure
```

## Structural result and model protocol

Unique-income matching covers **506,588 train rows (75.76%)** and **217,224 test rows (75.80%)**. Average agreement across twelve other covariates is **4.08321 / 4.08299** on train/test. Income-bin controls yield **3.78526 / 3.78539**; income-bin-plus-city controls yield **3.95157 / 3.94960**. The city-conditioned controls preserve the source-city match exactly.

Car-type agreement is **39.274% / 39.233%**, versus **33.517% / 33.530%** under the city-conditioned control. At least eight matching columns occurs in **1.217% / 1.196%** of eligible rows versus **0.916% / 0.903%** in controls. Agreement also increases for charging counts and range anxiety. These replicated associations motivate an experiment, not an ancestry claim or proof of useful label signal.

Append thirteen numeric source-context features to the round-seven encoder: unique-income coverage, agreement fraction, five categorical match flags, four numeric differences, source mechanism score and mechanism-score difference. They use public source covariates only, without competition labels or added source labels; the incumbent already uses source target means. Ambiguous source incomes have no assigned source row and zero-valued context with coverage flag zero. Missing source numeric values produce zero differences.

Screen on the existing ten-fold partition zero against **0.945084147**. Expand only if the gain is at least **0.00002**. All model settings remain unchanged; context features are appended after target encoding. Do not tune source-match weights on confirmation outcomes.

```bash
python -m src.source_context --model xgb --splits 10 --folds 0 --output artifacts/r10-source-context
```

The fold-zero screen reached **0.945114255**, a **0.000030107** improvement, meeting the expansion criterion. Expand the unchanged model to all ten folds. Before full results, declare the existing two-run blend grid against round seven (singles; probability weights 25/50/75%; equal rank weights). Both have the same outer folds. Submit only if selected mean fold AUC exceeds round seven by **0.00002** and pooled OOF also increases. This is a practical selection threshold, not statistical significance.

## Unlabeled robustness checks

After the initial structural result, an additional control conditions on source city **and car type** within income bins, preserving both their match rates exactly. Remaining agreement still exceeds this control: observed **4.08321 / 4.08299** versus null **4.01140 / 4.00820** on train/test. This post-hoc robustness check does not alter the model feature set or screen rule. Equal-weighting distinct matched income values is also reported, so the conclusion can be checked without a few frequent incomes dominating the statistic. These controls cannot rule out all distributional confounders.

## Completed model results

| Run | Pooled OOF AUC |
|---|---:|
| Round-seven incumbent | 0.946221346 |
| Source-context features | 0.946217521 |
| Best screened blend, equal rank weights | 0.946237878 |

The standalone model is slightly worse (**−0.000003825** pooled AUC). The best blend improves mean fold AUC by only **0.000012932**, below the predeclared **0.00002** submission threshold. Prediction correlation is **0.999775**. Do not submit this round or replace the public best (**0.94632**).

The unlabeled experiment supports multi-column source associations beyond the tested controls. It does not establish exact ancestry, recover a generating model, or demonstrate a new useful source of label information. Thirteen explicit context features did not materially improve the incumbent. Failure of this representation does not prove that every possible source-based method will fail.

## Verification

All ten fold artifacts passed the trainer's exact validation-index checks and cover the full training set. The saved estimator/encoder parameters and artifact hashes are exported in `runs/`. A regression test confirms ambiguous and unseen incomes receive no arbitrary source assignment and that training/inference context construction agrees. Tests: **13 passed**. Full saved-model test-set reproduction was not run because no new submission passed the selection criterion.
