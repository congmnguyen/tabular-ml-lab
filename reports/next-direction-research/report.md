# Next-direction research — 2026-09-09

## Recommendation

Prioritize a controlled neural representation experiment: **RealMLP with numerical features plus learned categorical embeddings for repeated exact income and commute values**. Evaluate its contribution to the existing XGBoost ensemble, not only standalone AUC. This is a hypothesis supported by an untested representation gap in our code and relevant primary literature; no new training or submission was performed during this research.

Best measured public result remains **0.94632**. The Kaggle CLI leaderboard checked during this review still lists Chris Deotte at **0.94672**. We did not find a verified public reproduction of that leading submission in the sources reviewed.

## What our experiments actually covered

- The old neural encoder in `src/neural.py` converts its selected inputs to one numeric matrix and quantile-transforms them. It includes low-cardinality category codes, raw numeric values, selected income digits, exact-value target means and source statistics. It does **not** give exact income/commute identities their own learned categorical embeddings. TabM receives `n_num_features`, with no categorical input. Linear-ReLU and piecewise numerical embeddings were tested.
- Native categorical CatBoost and XGBoost, sparse exact-value logistic regression, and joint additive income/commute effects were tested and reported. These are not equivalent to learned multidimensional value embeddings with nonlinear interactions.
- Recent tree variants correlate around 0.9992–0.9998 with the incumbent. The earlier neural model was less correlated but weaker; its small OOF blend gain did not increase the public score. A neural retry needs a specific representation change and a controlled blend check.
- Three fixed-budget holdouts, inference-support matching, and feature-family ablations failed to produce a robust full-OOF replacement. More versions of the same scalar encodings currently have weak support.

## Primary sources reviewed

### RealMLP paper and official implementation

[Holzmüller, Grinsztajn and Steinwart, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html) evaluates RealMLP and boosted-tree defaults across separate benchmark sets and reports benefits from combining the model families. Its benchmark range is 1K–500K rows; our 668K-row training set is larger, and the paper does not establish an EV competition score.

The [official PyTabKit repository](https://github.com/dholzmueller/pytabkit) supports categorical columns, numerical preprocessing, explicit validation inputs and configurable ensembles. Its example recommends disabling label smoothing for AUC/log-loss. Use this maintained implementation as the starting point, with dependency compatibility and runtime profiled first; do not launch its expensive HPO ensemble defaults blindly on an 8 GB GPU.

### Competition-specific RealMLP notebook

[Vladimir Demidov's RealMLP notebook](https://www.kaggle.com/code/yekenot/ps-s6-e9-realmlp-pytorch) provides a custom PyTorch implementation with learned categorical embeddings, periodic numerical embeddings, numerical values also exposed as categorical keys, and income/anxiety interactions. The downloaded revision sets **epochs=2** and contains no stored executed outputs from which we can verify its AUC. It also uses feature preprocessing before outer CV. Treat it as an idea source, not validated evidence of a winning score; implement fold-local preprocessing ourselves.

### TabM documentation

The [official TabM documentation](https://github.com/yandex-research/tabm) distinguishes categorical inputs, numerical embeddings and custom input modules. Its simple categorical API uses one-hot encoding, so high-cardinality income keys should not be assumed to become compact learned embeddings automatically. A custom embedding front end or RealMLP is needed to test the proposed representation directly. The old numerical-only TabM experiment does not exhaust this hypothesis.

### Other current competition notebooks

- [Pure LightGBM by Naji](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637): downloaded prose reports CV approximately 0.94607 and public 0.94638; the title contains different rounded values. Code uses multiscale bins, three target-encoder smoothing choices and a depth-five LightGBM. Scores are author-reported, not reproduced here. It overlaps substantially with experiments already run. A matched modern LightGBM baseline is a fallback, not the strongest new structural idea.
- [CTBoost by Markus.JM](https://www.kaggle.com/code/maiernator/s6e9-ctboost-not-catboost-astra-baseline): states historical CV 0.945958/public 0.94615, while the downloaded revision defaults to `RUN_CV=False`. These figures do not support prioritizing it over the measured incumbent. No CTBoost package was installed or executed.
- [Chovy's token transformer notebook](https://www.kaggle.com/code/chovyxu/ev-adoption-tokens-xgboost-tinytokentransformer): code includes tokenized numerical/categorical inputs, nested target encodings and several boosted-tree views. The downloaded artifact has no stored execution outputs establishing a score we can independently verify. It is a larger implementation surface than the proposed RealMLP control.
- [Competition discussion of the source mechanism](https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739142): discusses source reconstruction and mechanistic formulas, including disputed interpretation. Those are mostly already covered by our source recovery and offset experiments. Source reconstruction is not recovery of competition test labels or a proof of a competition AUC ceiling.

Notebook sources were downloaded and read only. No downloaded notebook code was executed, and no external prediction files were used. References and artifact SHA-256 values are saved in `references.json`.

## Bounded next experiment

1. Profile one official RealMLP configuration with one ensemble member on the available GPU. Check installed dependency compatibility before changing the working environment. Do not assume the notebook's eight-member setup fits comfortably.
2. Compare two input representations under the same architecture/training budget: ordinary continuous numerics with proper low-cardinality categories; and the same inputs plus exact income and commute categorical identities. Keep the original continuous values in both. Fit vocabularies/scalers on training rows only and provide an unknown-category fallback. Any target encoding must remain cross-fitted.
3. Use internal training data to set an epoch budget, then freeze it for the existing three repeated holdouts. Those holdouts are development checks, not pristine new tests. Report the embedding variant's gain against its neural control as well as its blend contribution against the cached XGBoost baseline.
4. Predeclare 0%, 10% and 20% neural probability-blend weights. A weaker standalone neural model is not automatically rejected if it reliably improves the blend. Low prediction correlation alone is also insufficient.
5. Expand at most one candidate to the incumbent's ten outer folds only if the blend improves all three screens with mean AUC gain above 0.00002. Score every fold, preserve aligned OOF predictions and verify model reload before considering a submission. This gate limits search; it is not a statistical significance test.

If this representation test fails, the next lower-cost fallback is a modern LightGBM run on the same ten-fold/feature protocol. Multi-seed XGBoost averaging can be considered for incremental variance reduction, but it is not the main hypothesis for a large gain. Further source reconstruction, small smoothing grids and repeated fold-zero feature searches are lower priority given our negative results.

No claim is made that RealMLP will beat 0.94672. The proposed experiment is valuable because it tests a concrete capability absent from our current neural pipeline, using evidence and controls that can falsify the hypothesis.
