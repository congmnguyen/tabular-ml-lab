# Round 3 — conditional value effects

Start: 2026-09-09. Current public best 0.94618; target 0.94672. Seven daily submissions remain. Continue selecting models with validation, not public-score probing.

Prior findings: the main remaining signal is tied to repeated synthetic values. Marginal target means can conflate income-specific effects with subsidy/anxiety/concern. The two existing tree models correlate at 0.99932, so meaningful diversity matters more than tiny changes to their weights.

Planned screens on the same fold 0:

1. Conditional categorical keys (income × subsidy/anxiety/home charging, income bins × concern, commute × home charging), with inner-cross-fitted smoothed target means. Test whether conditional effects explain residuals hidden by marginal encodings.
2. A sparse additive logistic model with one-hot exact numeric values and spline numeric effects. This estimates per-value effects jointly with the other covariates instead of using marginal target means. Compare a few regularization strengths, then test fixed blends with the current model.

Promising screens must be checked on remaining folds. Reusing a development fold and early stopping introduce selection optimism; full OOF scores are development estimates. Public submissions are reserved for candidates that improve OOF. No public prediction files or private solutions are used.
