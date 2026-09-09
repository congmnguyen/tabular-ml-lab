# Round 13 — official RealMLP and repeated-value embeddings

Starting public best: **0.94632**. Target: **0.94672**.

## Controlled experiment

Use official PyTabKit RealMLP-TD. Compare two feature views under the same architecture, training budget and seeds: original numerical/categorical features with recipe, worry and income modulo 1000; the same features plus exact income and commute identities as categorical columns. Preserve numerical views in both. Vocabulary fitting is training-only; unseen categorical values are represented as missing categories for the official encoder. No competition target encodings or public-source target means are included in either view, avoiding internal-validation ambiguity and isolating the added identity representation.

Profile compatibility, memory, runtime and unseen-category prediction on a small subset drawn only from the first split's training rows. Choose a shared fixed epoch budget based on that profiling; no heldout labels will select the budget. One architecture and one identity variant; no hyperparameter grid. Freeze the training configuration before scoring three repeated holdouts (seeds 17, 91, 2026), using the cached fixed-budget XGBoost baseline on exactly the same heldout rows.

Predeclare neural probability weights **0%, 10%, 20%** against XGBoost. Report both standalone neural scores and paired blend gains. Expand at most one candidate to full ten-fold OOF only if one fixed configuration and weight improves all three holdouts with mean AUC gain above **0.00002**. This is a practical search gate, not statistical significance. Reused overlapping holdouts are development evidence, not pristine tests.

Official references and research rationale: [previous research](../next-direction-research/report.md), [PyTabKit](https://github.com/dholzmueller/pytabkit). External notebook code and predictions are not used.

## Profile and frozen configuration

PyTabKit **1.7.3**, PyTorch Lightning **2.6.5**, TorchMetrics **1.9.0** installed without replacing the existing base packages. A first profile exposed incompatible mixed matmul precision settings in Torch 2.11 when our explicit `high` setting met the library's TF32 setting; removing our override resolved it. No library source was patched.

Both two-epoch, 20,000-row training-only profiles complete and predict finite probabilities for unseen income/commute categories. Full settings are in `profile.json`; short-profile timings are not reliable full-run runtime estimates. Based on resource profiling, freeze **64 epochs**, batch **1024**, prediction batch **4096**, one ensemble member, model seed **42**, official architecture/default optimizer settings, and **label smoothing disabled**. Set validation fraction to zero and stop epoch to 64, so all outer-training rows are used and no scored holdout selects a checkpoint. Both input views have the same epoch budget. This reduced budget relative to the library's 256-epoch default limits the scope of any negative conclusion.

```bash
python -m src.profile_realmlp
python -m src.realmlp_audit
```

## Results

| Holdout seed | Fixed XGBoost | RealMLP continuous | RealMLP + identities |
| --- | ---: | ---: | ---: |
| 17 | 0.94603607 | 0.93912787 | 0.93291757 |
| 91 | 0.94557589 | 0.93890086 | 0.92866136 |
| 2026 | 0.94830078 | 0.94128952 | 0.93390346 |

| Neural view | Probability blend weight | Mean paired AUC change | Positive holdouts |
| --- | ---: | ---: | ---: |
| continuous | 10% | -0.00010217 | 0/3 |
| continuous | 20% | -0.00032696 | 0/3 |
| identities | 10% | -0.00017962 | 0/3 |
| identities | 20% | -0.00065728 | 0/3 |

All four predeclared blends lose AUC on all three holdouts. The identity view also loses standalone AUC against the continuous control on every holdout. No configuration passes the expansion gate: no ten-fold expansion and no Kaggle submission. Best public result remains **0.94632**, below the **0.94672** target.

These are paired repeated-holdout results, not pooled ten-fold OOF. The negative result applies to the fixed 64-epoch, single-member, no-target-encoding setup. It does not establish that a longer-trained or differently regularized neural model cannot help, and no such improvement has been demonstrated here.

## Identity coverage

A post-run, label-free coverage check finds that on seed 17 only **0.6102%** of heldout income values and **0.0090%** of commute values are unseen in training. Median training support is 125 rows per heldout income value and 1,213 per commute value. Training cardinalities are 12,821 and 799. Thus unseen identities are uncommon; this check alone cannot identify the cause of the neural deficit. Results for all three splits are in `identity-support.json`.

## Artifacts and validation

Each of the six fits saves its fold-local vocabulary, official model, heldout indices/targets/predictions, metrics, settings and code hashes under ignored `artifacts/r13-realmlp`. Aggregate scores are in `realmlp.json`; package versions are in `environment.json`. The optional dependencies are pinned in `requirements-realmlp.txt`.

The offline suite passes **15 tests**, including preservation of the numeric views and train-only identity vocabulary with unseen categories. Existing scikit-learn TargetEncoder deprecation warnings remain. GPU profiling and model reproduction are separate manual checks; ordinary CI does not install the optional RealMLP stack.

To repeat the saved-model verification:

```bash
python -m src.verify_realmlp
```

All six saved models were reloaded and predicted every heldout row (**66,867 rows per model**) on GPU with the original prediction batch size. Maximum absolute prediction difference was **0.0** for every model. `reproduction.json` records artifact/data hashes, effective interface configuration and official library defaults.
