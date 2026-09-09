# ML Template

A reproducible binary tabular classification pipeline, demonstrated on **Kaggle Playground S6E9 — Predicting Electric Vehicle Purchases**.

The case study compares linear, boosting and neural models on identical folds, investigates the synthetic data source, and records OOF metrics, error analysis and Kaggle submissions. The focus is traceable experiments and reliable preprocessing.

## Results

**Best submitted public AUC: 0.94632**, from the average of ten outer-fold XGBoost models with fold-fitted probit offsets, joint remainder features and target-encoding smoothing 2/20. Development pooled OOF AUC is 0.946221. The requested leaderboard target of 0.94672 has not been exceeded.

[Round 4 report](reports/round4/report.md) documents the mechanism audit, conditional permutation diagnostics, failed risk-adjusted encoding, five-fold improvements and the new public best. [Round 6 selected blend](reports/round6/selected.json). The [round 5 diagnostic](reports/round5/report.md) checks whether income-specific subgroup errors replicate on two held-out sets; the evidence did not justify adding conditional encodings.

[Round 2 report](reports/round2/report.md) documents expert sources, independent data investigation, cross-fitted target encoding, screening failures, and the submitted blend. [Selected blend](reports/round2/selected.json). The [round 3 report](reports/round3/report.md) adds nine run configurations and a five-fold TabM ensemble: development mean AUC increased to 0.946066, while its public submission tied 0.94618. These earlier blends are retained as historical baselines.

The initial pipeline below remains a simpler, inductive baseline: LightGBM with charging/commute features scored 0.94160 publicly; Logistic Regression scored 0.93738. Its original five configurations and 25 fold fits are preserved.

See the [experiment report](reports/ev-purchases.md), [predefined experiment plan](reports/experiment-plan.md), and [submission record](reports/submissions.md). Metrics come from executed runs; the competition's final private leaderboard is not yet available.

## Quickstart

Python 3.12 and a CPU are sufficient for the initial pipeline. Experiments ran on Linux with 22 GiB RAM; later XGBoost, native CatBoost and TabM experiments also used an RTX 4060. Base dependencies are pinned in `requirements.txt`; optional neural dependencies are in `requirements-neural.txt`.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
make test
```

Join the [competition](https://www.kaggle.com/competitions/playground-series-s6e9) and accept its rules, then configure Kaggle CLI credentials outside this repository. Install the CLI separately with `uv tool install kaggle`.

```bash
mkdir -p data/ev-purchases
kaggle competitions download playground-series-s6e9 -p data/ev-purchases
unzip data/ev-purchases/playground-series-s6e9.zip -d data/ev-purchases
make baseline DATA=data/ev-purchases
make experiments DATA=data/ev-purchases
make report DATA=data/ev-purchases
```

Each experiment refuses to overwrite a completed run. For a new experiment, use a new `--output` directory via the CLI. `make experiments` runs the four non-logistic candidates sequentially.

## Initial baseline pipeline

```text
train/test CSV + config
  → exclude ID/target, optional stateless features
  → fixed stratified folds (seed 42)
  → fit preprocessing and model within each train fold
  → validation probabilities → OOF metrics and error analysis
  → persist complete fold pipelines → average test probabilities
  → validate ID order and probability range → submission.csv
```

- Logistic Regression: train-fold median imputation/scaling and one-hot categorical encoding; unseen categories are ignored.
- LightGBM: train-fold imputation and ordinal categorical encoding; unseen categories receive -1. This is an ordinal baseline, not native LightGBM categorical processing.
- CatBoost: native categorical handling, explicit missing-category token, and numeric missing values.
- Fixed iteration budgets, without early stopping on the scored validation fold.
- Reloaded artifact predictions are checked against in-memory predictions for each fold.

## Initial baseline artifacts

Each ignored `artifacts/<run>/` contains:

| File | Purpose |
|---|---|
| `fold-N.joblib` | Model, preprocessing, feature schema and config |
| `oof.csv` | Original ID, binary target, fold and validation probability |
| `submission.csv` | Test IDs in input order and predicted probabilities |
| `metrics.json` | Fold/OOF metrics, runtime, config, source/data/fold hashes and Git revision |

Only load trusted joblib artifacts. Aggregate results and plots are committed under `reports/`; row-level data and model binaries stay local.

Predict from saved folds:

```bash
.venv/bin/python -m src.predict \
  --run artifacts/lightgbm-features \
  --test data/ev-purchases/test.csv \
  --output artifacts/reproduced-submission.csv
```

Submit explicitly after checking the selected run in `reports/selected.json`:

```bash
kaggle competitions submit playground-series-s6e9 \
  -f artifacts/lightgbm-features/submission.csv \
  -m "Five-fold LightGBM with charging/commute features; selected by CV"
kaggle competitions submissions playground-series-s6e9
```

## Expert-informed competition experiments

These EV-specific experiments extend the generic baseline. They use public original-data statistics and internally cross-fitted target encoders, with early stopping on each scored outer fold. Scores are development estimates with selection optimism. The selected blend's frequency maps use outer training features only; the separately screened transductive variant was not selected.

```bash
kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety \
  -p ~/.cache/kaggle/ev-original --unzip
.venv/bin/python -m src.expert --output artifacts/new-full-lgb --model lgb --mode full --rounds 6000
.venv/bin/python -m src.expert --output artifacts/new-full-xgb --model xgb --mode full --rounds 5000
.venv/bin/python -m src.blend_experts \
  --runs artifacts/new-full-lgb artifacts/new-full-xgb --output artifacts/new-blend
.venv/bin/python -m src.predict_expert --run artifacts/new-blend \
  --test data/ev-purchases/test.csv --output artifacts/new-reproduced.csv
```

The expert runner defaults to competition CSVs under `~/.cache/kaggle/playground-series-s6e9`; pass `--data data/ev-purchases` for another directory and `--original path/to/original.csv` when needed. XGBoost is configured for an NVIDIA CUDA GPU; LightGBM uses CPU. The other screening modules and their applied settings are documented in the round-two report and run records. `--folds 0` performs a screen; omitted `--folds` trains all five. Completed folds are reused, so use a new output directory for changed settings.

Rank-blend outputs are scores for ROC-AUC, not calibrated purchase probabilities. Expert artifacts use `foldN.joblib`, `foldN.npz`, and `summary.json`; the baseline runner retains its separate artifact format.

## Reuse and scope

For another binary tabular dataset, provide `train.csv` and `test.csv`, then change the ID, target and positive label in a copied JSON config. Pass `--config path/to/config.json`; leave `--engineered` off because those features are EV-specific. Both files must have the same feature schema and order. Targets must have exactly two non-null classes.

Random stratification assumes exchangeable rows. Group splits, time splits, regression, production serving and calibrated decision thresholds are not implemented. This template has one completed competition case study; generality across multiple datasets is not yet demonstrated.

The original learning code based on Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem* is preserved in `legacy/` for attribution/history. Its old runner and dependencies are not used by the current pipeline. The original repository also retains its exploratory notebook.

## Tests

`make test` verifies train-only preprocessing, unseen categories, feature schema checks, artifact round trips, submission integrity, and a small end-to-end run repeated to verify identical folds and predictions. GitHub Actions runs these offline tests without Kaggle credentials.

## License and data

Code: [MIT](LICENSE). Competition data remains subject to [Kaggle competition rules](https://www.kaggle.com/competitions/playground-series-s6e9/rules); it is not redistributed here.

[Round 6 report](reports/round6/report.md): public notebook hypotheses independently checked, source-feature recovery, smoothing experiments and exact saved-model reproduction. Selected blend public AUC: **0.94627**.

[Round 7 report](reports/round7/report.md): ten outer folds improve pooled OOF and public AUC (**0.94632**); all predictions reproduced exactly from saved models. [Current selected run](reports/round7/runs/r7-tenfold.json).

[Round 8 diagnostic](reports/round8/report.md): broad shared error regions replicate partly, but a fixed holdout correction adds only 0.000011 AUC; no new submission.
