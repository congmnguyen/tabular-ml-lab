# Round 11 — stability and target-statistic support

Starting best public AUC: **0.94632**; target **0.94672**.

## Fixed-budget stability protocol

Compare the round-seven baseline, round-nine joint effects, and round-ten source context on three repeated stratified 90/10 holdouts (seeds **17, 91, 2026**). Every comparison within a seed uses exactly the same rows, model seed 42, and **600 trees**, with no early stopping or tuning on the scored holdout. Retain five inner target-encoding folds and smoothing 2/20. All choices are set before the new results.

These are overlapping development holdouts over data used previously. They assess sensitivity to partitions and early-stopping choices; they are not untouched tests or independent statistical replicates. No leaderboard submission will be made from partial holdouts. Report all signed paired differences, not only the average or favorable seeds.

## Target-encoding diagnostic, planned next

Use the frozen baseline models from the above comparisons. On each heldout set, compare ordinary target encodings based on 100% of outer training rows against encodings fitted on 80% subsets of outer training rows, matching the support fraction used by inner cross-fitting. Average predictions across five predetermined inner-fold encoders. Holdout labels never fit either the encoders or base models. This tests inference-time support matching plus averaging, not pure support alone. It sacrifices some information per encoder; improvement is not assumed.

Measure encoding changes by income frequency, as well as AUC and log loss. Do not tune subset sizes, seeds, or weights against these outcomes. Only a positive mean AUC gain above **0.00002** and positive gains on all three splits would justify a full OOF experiment; this practical gate is not a significance test.

```bash
python -m src.stability_audit
```

## Stability results

| Holdout seed | Joint-effect AUC difference | Source-context AUC difference |
|---|---:|---:|
| 17 | −0.000000657 | +0.000058083 |
| 91 | −0.000051902 | +0.000040622 |
| 2026 | −0.000066901 | −0.000009462 |

Joint effects lose on all three fixed-budget splits (mean **−0.000039820**). Source context wins on two and loses slightly on one (mean **+0.000029748**). These checks weaken the case for the standalone joint-effect candidate and leave source context promising but inconsistent. They do not test the previously submitted joint-effect ensemble and do not establish final-test superiority for source context.

The paired comparison controls training budget within each split. Differences from earlier ten-fold experiments may reflect both changed partitions and fixed versus early-stopped training; this experiment does not isolate those factors individually.

## Encoding-support results

| Holdout seed | AUC change from subset-statistic prediction averaging |
|---|---:|
| 17 | −0.000014078 |
| 91 | +0.000020415 |
| 2026 | +0.000043868 |

Mean gain is **0.000016735**. This misses both parts of the predeclared expansion gate: it is below 0.00002 and one split declines. Do not expand to full OOF or submit. The selected pipeline remains unchanged, public best **0.94632**.

For incomes seen 1–4 times in outer training, the mean absolute change of the low-smoothing income target statistic is approximately **0.027** when using an 80% subset instead of all outer-training rows. For incomes seen at least 100 times it is approximately **0.009**. These values average absolute changes across the five subset encoders, not the shift of their averaged statistic. Resulting averaged predictions change by about **0.0045–0.0050** and **0.0017–0.0022**, respectively. This confirms greater sensitivity among rare values; it does not show that changing their encodings improves ranking.

The new experiment adjusts inference statistics only. It does not change how the model was trained, and averaging five subset predictions also changes variance. Therefore the result does not isolate a universal train/inference-mismatch penalty or rule out every other encoding design.

```bash
python -m src.encoding_support_audit
```

## Decision and limits

No new submission. We now have evidence that the recent small gains depend on partition and training protocol: standalone joint effects do not survive this check, source context is mixed, and inference-support matching is mixed. These results favor requiring repeated improvements before spending more submissions, rather than treating each favorable fold as a new best. They do not prove an AUC ceiling.

## Verification

Reloaded all nine saved model/encoder bundles and compared 512 held-out rows per model to stored predictions: exact agreement, maximum difference **0.0**. This is a sample reload check, not full test-set reproduction. Actual estimator and target-encoder parameters, data hashes and code hashes are recorded in `provenance.json`. Existing tests: **13 passed**.
