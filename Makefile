PYTHON ?= .venv/bin/python
DATA ?= $(HOME)/.cache/kaggle/playground-series-s6e9

.PHONY: test baseline experiments report

test:
	$(PYTHON) -m pytest -q

baseline:
	$(PYTHON) -m src.pipeline --data "$(DATA)" --output artifacts/logistic --model logistic

experiments:
	$(PYTHON) -m src.pipeline --data "$(DATA)" --output artifacts/lightgbm --model lightgbm
	$(PYTHON) -m src.pipeline --data "$(DATA)" --output artifacts/catboost --model catboost
	$(PYTHON) -m src.pipeline --data "$(DATA)" --output artifacts/lightgbm-regularized --model lightgbm --variant
	$(PYTHON) -m src.pipeline --data "$(DATA)" --output artifacts/lightgbm-features --model lightgbm --engineered

report:
	$(PYTHON) -m src.analyze --data "$(DATA)"
