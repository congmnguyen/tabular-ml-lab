"""Ensure inference respects saved fold count and old five-fold artifacts."""
import json
import numpy as np
import pandas as pd
import pytest
from src.predict_expert import predict_run


@pytest.mark.parametrize('splits,metadata', [(5, {}), (10, {'splits': 10})])
def test_predict_uses_saved_partition_count(tmp_path, monkeypatch, splits, metadata):
    (tmp_path / 'summary.json').write_text(json.dumps({'args': metadata}))
    seen = []

    class Encoder:
        def transform(self, frame):
            return frame

    class Model:
        def __init__(self, value):
            self.value = value

        def predict_proba(self, frame):
            p = np.full(len(frame), self.value, dtype=np.float32)
            return np.column_stack([1-p, p])

    def load(path):
        fold = int(path.stem.removeprefix('fold'))
        seen.append(fold)
        return {'encoder': Encoder(), 'model': Model((fold+1)/(splits+1))}

    monkeypatch.setattr('src.predict_expert.joblib.load', load)
    result = predict_run(tmp_path, pd.DataFrame({'id': [1, 2]}))
    assert seen == list(range(splits))
    np.testing.assert_allclose(result, .5, atol=1e-7)
