# Round 12 — feature-family ablation

Starting best public AUC: **0.94632**. Target: **0.94672**.

## Predeclared protocol

Use the same repeated stratified 90/10 holdouts (seeds 17, 91, 2026), training rows, five inner encoding folds and fixed 600-tree XGBoost parameters as round eleven. Recreate training encodings with `fit_transform`; never use full-fit `transform` to encode training targets. Compare five removals individually against the saved baseline:

1. All negative-power digit features and their derived target/frequency encodings.
2. Public-source target means.
3. Frequency features.
4. Low-smoothing target encodings (smoothing 2).
5. High-smoothing target encodings (smoothing 20).

The base feature matrix is held fixed within each seed; only the selected family is dropped before fitting a fresh model. The fold-fitted probit mechanism and its helper columns remain. No early stopping, no holdout-driven parameter tuning, no combining removal groups after seeing these results.

A candidate must improve all three splits and have mean AUC gain above 0.00002 to justify full OOF expansion. Five comparisons over reused development holdouts still create selection optimism; this practical gate is not statistical significance. Do not submit partial-holdout predictions.

```bash
python -m src.ablation_audit
```

If multiple removals pass the three-split gate, expand only the one with the highest mean gain. This tie-breaking rule is recorded while the third split is still pending. Full OOF expansion uses the incumbent's existing ten-fold seed-42 protocol and early-stopping budget, so the final candidate is compared under the same protocol as the current best; its scored-fold stopping remains a source of development optimism. Do not combine removal groups in this round.

## Repeated-holdout results

| Removed family | Mean AUC change | Wins / 3 | Passes gate |
|---|---:|---:|---|
| Negative-power digits and derived encodings | +0.000035087 | 2 | No |
| Public-source target means | +0.000030653 | 3 | **Yes** |
| Frequencies | +0.000010023 | 2 | No |
| Low-smoothing encodings | −0.000044061 | 0 | No |
| High-smoothing encodings | +0.000024811 | 2 | No |

Only removing public-source target means passes the gate, so it is expanded. The larger average gain from deleting negative-power digits does not override its failure to improve all three splits. Results are available in `ablation.json`. These comparisons suggest some source target means may add noise under this protocol; they do not prove all public-source information is harmful.

```bash
python -m src.no_source_means --model xgb --splits 10 --output artifacts/r12-no-source-means
```

For the full OOF result, compare the selected candidate alone and the incumbent/candidate fixed two-run blend grid (probability weights 25/50/75%, equal rank weights). Submit only if mean fold AUC improves by more than 0.00002 and pooled OOF also improves. No other removal groups will be mixed in.

## Full OOF result and decision

| Run | Pooled OOF AUC |
|---|---:|
| Incumbent | 0.946221346 |
| Remove source target means | 0.946213915 |
| Best fixed-grid blend, equal rank weights | 0.946239166 |

Removing source target means alone slightly reduces pooled OOF AUC (**−0.000007431**). The best blend gains only **0.000014314** mean fold AUC, below the **0.00002** submission threshold. Prediction correlation is **0.999720**. Do not submit or replace the incumbent; best public AUC remains **0.94632**, target **0.94672** unmet.

The promising repeated-holdout result did not carry over to standalone full OOF. The full run uses the incumbent's early-stopping protocol, while the preliminary screen used fixed 600-tree fits; both partition and training-protocol differences can contribute. This does not prove source means universally help or hurt. It demonstrates why a small development screen should not directly trigger a submission.

## Verification

All fifteen ablation model bundles were reloaded and checked on 256 stored validation rows each; predictions matched exactly. Their actual parameters, dropped columns and hashes are in `ablation-provenance.json`. Full-run fold artifacts passed exact validation-index checks. The ten full-run models were also checked on 256 stored OOF rows each, with no source-mean feature present and exact prediction agreement. These are sample checks, not complete test-set reproduction; no new submission passed the selection criterion. The source-label-invariance regression test confirms that flipping public source labels cannot affect the selected encoder's output. Existing tests: **14 passed**.
