"""Reproducible binary tabular experiments; preprocessing learns from train folds only."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(frame, config, engineered=False):
    x = frame.drop(columns=[config['id'], config['target']], errors='ignore').copy()
    if engineered:
        # Stateless, domain-motivated features: charging access and commute burden.
        x['Charging_Total'] = x.Charging_Stations_Near_Home + x.Charging_Stations_Near_Work
        x['Commute_Per_Car'] = x.Daily_Commute_km / (1 + x.Number_of_Cars_Owned)
        x['Log_Income'] = np.log1p(x.Annual_Income_USD.clip(lower=0))
        x['Home_Access_Commute'] = x.Daily_Commute_km * x.Home_Charging_Possible.eq('Yes')
    return x


def build_model(x, kind, config, variant=False):
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in x if c not in numeric]
    numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scale', StandardScaler())])
    if kind == 'logistic':
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    else:
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                                 ('encode', cat_encoder)])
    pre = ColumnTransformer([('numeric', numeric_pipe, numeric), ('categorical', categorical_pipe, categorical)])
    seed, threads = config['seed'], config['threads']
    if kind == 'logistic':
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=seed)
    elif kind == 'lightgbm':
        model = LGBMClassifier(n_estimators=650 if variant else 350,
                              learning_rate=.035 if variant else .05,
                              num_leaves=15 if variant else 31,
                              min_child_samples=100 if variant else 50,
                              reg_lambda=5 if variant else 1,
                              verbosity=-1, n_jobs=threads, random_state=seed,
                              deterministic=True, force_col_wise=True)
    else:
        model = CatBoostClassifier(iterations=500, depth=6, learning_rate=.07,
                                   loss_function='Logloss', random_seed=seed,
                                   thread_count=threads, verbose=False, allow_writing_files=False)
        # CatBoost gets strings and native categorical handling; no learned global encoding.
        return model, categorical
    return Pipeline([('preprocess', pre), ('model', model)]), categorical


def cat_frame(x, categorical):
    x = x.copy()
    for c in categorical:
        x[c] = x[c].fillna('__MISSING__').astype(str)
    for c in x.columns.difference(categorical):
        x[c] = x[c].astype(float)
    return x


def score(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'log_loss': float(log_loss(y, p)),
            'average_precision': float(average_precision_score(y, p))}


def validate_submission(frame, test, config):
    assert list(frame.columns) == [config['id'], config['target']]
    assert frame[config['id']].equals(test[config['id']])
    assert frame[config['id']].is_unique
    p = frame[config['target']].to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def predict_bundle(bundle, frame):
    x = features(frame, bundle['config'], bundle['engineered'])
    if list(x.columns) != bundle['columns']:
        raise ValueError('Feature schema/order differs from training')
    if bundle['kind'] == 'catboost':
        x = cat_frame(x, bundle['categorical'])
    return bundle['model'].predict_proba(x)[:, 1]


def run(args):
    config = json.loads(Path(args.config).read_text())
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError('Completed run exists; choose a new output directory')
    train, test = pd.read_csv(data/'train.csv'), pd.read_csv(data/'test.csv')
    if train[config['target']].isna().any() or train[config['target']].nunique() != 2:
        raise ValueError('Expected a non-null binary target')
    if config['positive_label'] not in train[config['target']].unique():
        raise ValueError('Positive label absent')
    assert train[config['id']].is_unique and test[config['id']].is_unique
    y = train[config['target']].eq(config['positive_label']).astype(int)
    x = features(train, config, args.engineered)
    xt = features(test, config, args.engineered)
    if list(x.columns) != list(xt.columns):
        raise ValueError('Train/test feature schema differs')
    folds = np.full(len(train), -1, dtype=int)
    oof = np.full(len(train), np.nan)
    predictions = np.zeros(len(test))
    results = []
    start = time.monotonic()
    splitter = StratifiedKFold(config['folds'], shuffle=True, random_state=config['seed'])
    for fold, (tr, va) in enumerate(splitter.split(x, y)):
        tick = time.monotonic()
        model, cats = build_model(x, args.model, config, args.variant)
        if args.model == 'catboost':
            model.fit(cat_frame(x.iloc[tr], cats), y.iloc[tr], cat_features=cats)
        else:
            model.fit(x.iloc[tr], y.iloc[tr])
        bundle = {'model': model, 'kind': args.model, 'categorical': cats,
                  'columns': list(x.columns), 'config': config, 'engineered': args.engineered}
        oof[va] = predict_bundle(bundle, train.iloc[va])
        predictions += predict_bundle(bundle, test) / config['folds']
        folds[va] = fold
        path = out/f'fold-{fold}.joblib'
        joblib.dump(bundle, path)
        np.testing.assert_allclose(predict_bundle(joblib.load(path), train.iloc[va[:128]]), oof[va[:128]], rtol=1e-12, atol=1e-12)
        result = {'fold': fold, **score(y.iloc[va], oof[va]), 'seconds': time.monotonic()-tick}
        results.append(result)
        print(json.dumps(result), flush=True)
    assert np.isfinite(oof).all() and (folds >= 0).all()
    pd.DataFrame({'id': train[config['id']], 'target': y, 'fold': folds, 'prediction': oof}).to_csv(out/'oof.csv', index=False)
    submission = pd.DataFrame({config['id']: test[config['id']], config['target']: predictions})
    validate_submission(submission, test, config)
    submission.to_csv(out/'submission.csv', index=False)
    summary = {'model': args.model, 'engineered': args.engineered, 'variant': args.variant,
               'config': config, 'rows': len(train), 'test_rows': len(test),
               'positive_rate': float(y.mean()), 'folds': results, 'oof': score(y, oof),
               'auc_mean': float(np.mean([r['auc'] for r in results])),
               'auc_std': float(np.std([r['auc'] for r in results])),
               'seconds': time.monotonic()-start, 'python': platform.python_version(),
               'train_sha256': digest(data/'train.csv'), 'test_sha256': digest(data/'test.csv'),
               'fold_sha256': hashlib.sha256(folds.tobytes()).hexdigest(),
               'source_sha256': digest(__file__),
               'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()}
    (out/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/ev-purchases/config.json')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['logistic','lightgbm','catboost'], required=True)
    parser.add_argument('--engineered', action='store_true')
    parser.add_argument('--variant', action='store_true')
    run(parser.parse_args())

if __name__ == '__main__':
    main()
