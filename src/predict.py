"""Predict with saved fold pipelines, preserving test ID order."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .pipeline import predict_bundle, validate_submission

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    paths = sorted(Path(a.run).glob('fold-*.joblib'))
    if not paths:
        raise ValueError('No saved folds found')
    test = pd.read_csv(a.test)
    predictions = []
    for path in paths:
        bundle = joblib.load(path)
        predictions.append(predict_bundle(bundle, test))
    config = bundle['config']
    if len(paths) != config['folds']:
        raise ValueError('Incomplete fold artifacts')
    sub = pd.DataFrame({config['id']: test[config['id']], config['target']: np.mean(predictions, axis=0)})
    validate_submission(sub, test, config)
    sub.to_csv(a.output, index=False)
