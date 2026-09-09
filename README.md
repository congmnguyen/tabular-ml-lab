# ML Template

A reproducible binary tabular classification pipeline, demonstrated on **Kaggle Playground S6E9 — Predicting Electric Vehicle Purchases**.

The case study compares Logistic Regression, LightGBM and CatBoost on identical folds, tests two feature/model hypotheses, and records OOF metrics, error analysis and Kaggle submissions. The focus is traceable experiments and reliable preprocessing.

## Results

**Selected model: LightGBM with charging/commute features — mean 5-fold AUC 0.941800, public Kaggle AUC 0.94160.** Logistic baseline public AUC: 0.93738. Five configurations and 25 fold fits completed.

See the [experiment report](reports/ev-purchases.md), [predefined experiment plan](reports/experiment-plan.md), and [submission record](reports/submissions.md). Metrics come from executed runs; the competition's final private leaderboard is not yet available.

## Quickstart

Python 3.12 and a CPU are sufficient. The completed case study used eight threads per model on a Linux machine with 22 GiB RAM. Dependencies are pinned; the GPU was not used.

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

## Pipeline

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

## Saved artifacts

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

## Reuse and scope

For another binary tabular dataset, provide `train.csv` and `test.csv`, then change the ID, target and positive label in a copied JSON config. Pass `--config path/to/config.json`; leave `--engineered` off because those features are EV-specific. Both files must have the same feature schema and order. Targets must have exactly two non-null classes.

Random stratification assumes exchangeable rows. Group splits, time splits, regression, production serving and calibrated decision thresholds are not implemented. This template has one completed competition case study; generality across multiple datasets is not yet demonstrated.

The original learning code based on Abhishek Thakur's *Approaching (Almost) Any Machine Learning Problem* is preserved in `legacy/` for attribution/history. Its old runner and dependencies are not used by the current pipeline. The original repository also retains its exploratory notebook.

## Tests

`make test` verifies train-only preprocessing, unseen categories, feature schema checks, artifact round trips, submission integrity, and a small end-to-end run repeated to verify identical folds and predictions. GitHub Actions runs these offline tests without Kaggle credentials.

## License and data

Code: [MIT](LICENSE). Competition data remains subject to [Kaggle competition rules](https://www.kaggle.com/competitions/playground-series-s6e9/rules); it is not redistributed here.
