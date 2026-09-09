"""Reload every RealMLP holdout model and verify its saved predictions."""
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .investigate import DATA
from .pipeline import digest


def main():
    import torch
    from pytabkit.models.sklearn.default_params import DefaultParams

    frame = pd.read_csv(DATA / 'train.csv')
    root = Path('artifacts/r13-realmlp')
    records = []
    for path in sorted(root.glob('*.joblib')):
        bundle = joblib.load(path)
        saved = np.load(path.with_suffix('.npz'))
        features = bundle['features'].transform(frame.iloc[saved['indices']])
        model = bundle['model']
        model.to('cuda')
        prediction = model.predict_proba(features)[:, 1]
        difference = float(np.max(np.abs(prediction - saved['prediction'])))
        assert np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction, saved['prediction'], atol=1e-7, rtol=0)
        records.append({
            'model': path.name,
            'rows': len(prediction),
            'max_absolute_difference': difference,
            'model_sha256': digest(path),
            'predictions_sha256': digest(path.with_suffix('.npz')),
            'effective_config': model.alg_interface_.config,
        })
        del bundle, model, features
        gc.collect()
        torch.cuda.empty_cache()
    assert len(records) == 6, f'Expected six completed models, found {len(records)}'
    report = {
        'verification': records,
        'library_defaults': DefaultParams.RealMLP_TD_CLASS,
        'train_sha256': digest(DATA / 'train.csv'),
        'script_sha256': digest(__file__),
    }
    destination = Path('reports/round13/reproduction.json')
    destination.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps([{
        k: r[k] for k in ('model', 'rows', 'max_absolute_difference')
    } for r in records], indent=2))


if __name__ == '__main__':
    main()
