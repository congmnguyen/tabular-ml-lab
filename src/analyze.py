"""Build aggregate report and plots from actual OOF predictions; no raw rows published."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--report', default='reports')
    a = parser.parse_args()
    out = Path(a.report)
    out.mkdir(exist_ok=True, parents=True)
    train = pd.read_csv(Path(a.data)/'train.csv')
    test = pd.read_csv(Path(a.data)/'test.csv')
    runs = {p.parent.name: json.loads(p.read_text()) for p in Path(a.artifacts).glob('*/metrics.json')}
    if len(runs) < 5:
        raise ValueError('Expected all five planned experiments')
    assert len({r['fold_sha256'] for r in runs.values()}) == 1
    assert len({r['train_sha256'] for r in runs.values()}) == 1
    winner = max(runs, key=lambda name: runs[name]['auc_mean'])
    (out/'selected.json').write_text(json.dumps({'run':winner, 'rule':'maximum mean five-fold ROC-AUC', 'metrics':runs[winner]}, indent=2)+'\n')
    metrics_dir = out/'metrics'
    metrics_dir.mkdir(exist_ok=True)
    for name, result in runs.items():
        (metrics_dir/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
    best = pd.read_csv(Path(a.artifacts)/winner/'oof.csv')
    assert best.id.equals(train.id)
    y, p = best.target, best.prediction
    data_summary = {
        'train_rows':len(train), 'test_rows':len(test), 'features':len(train.columns)-2,
        'positive_rate':float(y.mean()), 'missing_train':train.isna().sum().to_dict(),
        'missing_test':test.isna().sum().to_dict(),
        'duplicate_train_features':int(train.drop(columns=['id','Will_Buy_EV']).duplicated().sum()),
        'overlapping_ids':len(set(train.id)&set(test.id)),
        'target_counts':train.Will_Buy_EV.value_counts().to_dict(),
    }
    (out/'data-summary.json').write_text(json.dumps(data_summary, indent=2)+'\n')
    lines = ['# EV purchase prediction — experiment report', '',
             'Run date: 2026-09-09. Synthetic Kaggle Playground S6E9 data; this is a competition case study, not a validated real-world purchase forecasting system.', '',
             f'Train: {len(train):,} rows. Test: {len(test):,} rows. Positive class: Yes ({y.mean():.2%}). IDs are excluded from features. See [data checks](data-summary.json).', '',
             '## Validation and selection', '',
             'All runs use the same five stratified folds and seed 42. Numeric imputation/scaling and categorical encoding are fitted within each training fold. CatBoost uses native categorical features. Test predictions average the five saved models. Every fold artifact was reloaded and its predictions checked against the in-memory model.', '',
             'Selection uses mean fold ROC-AUC; leaderboard scores do not select parameters. Fixed iteration budgets avoid early-stopping selection on the evaluation fold. Repeated development on the same folds still introduces selection optimism; there is no independent labeled final holdout.', '',
             '## Results', '', '| Run | Mean fold AUC | Fold SD | OOF AUC | Log loss | AP | Wall seconds |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in sorted(runs.items(), key=lambda item:-item[1]['auc_mean']):
        lines.append(f"| {name} | {r['auc_mean']:.6f} | {r['auc_std']:.6f} | {r['oof']['auc']:.6f} | {r['oof']['log_loss']:.6f} | {r['oof']['average_precision']:.6f} | {r['seconds']:.1f} |")
    lines += ['',f'**Selected: {winner}.** The fold standard deviation is descriptive, not a confidence interval. Timings include validation/test prediction and artifact checks; runs overlapped on the same machine and are not controlled speed benchmarks.', '',
              '![CV comparison](cv-comparison.png)', '', '## Improvement experiments', '']
    for name in ['lightgbm-regularized','lightgbm-features']:
        delta = runs[name]['auc_mean']-runs['lightgbm']['auc_mean']
        paired = [v['auc']-b['auc'] for v,b in zip(runs[name]['folds'], runs['lightgbm']['folds'])]
        lines.append(f"- {name}: mean AUC delta {delta:+.6f} vs LightGBM baseline; improved {sum(d>0 for d in paired)}/5 folds. Paired fold deltas: {', '.join(f'{d:+.6f}' for d in paired)}.")
    lines += ['', '## Error analysis', '',
              'The following diagnostics use selected-model OOF probabilities. A threshold of 0.5 is used only to describe errors, not as an optimized business decision rule.', '',
              f'Confusion matrix [[TN, FP], [FN, TP]] at 0.5: `{confusion_matrix(y,p >= .5).tolist()}`.', '',
              '| Segment | Value | Rows | Positives | Positive rate | AUC | Log loss |', '|---|---|---:|---:|---:|---:|---:|']
    segments=[]
    for col in ['City_Type','Gender','Home_Charging_Possible','Range_Anxiety_Level']:
        for value, group in train.groupby(col, dropna=False):
            ix = group.index
            auc = roc_auc_score(y.iloc[ix], p.iloc[ix]) if y.iloc[ix].nunique()==2 else float('nan')
            row = dict(segment=col,value=str(value),rows=len(ix),positives=int(y.iloc[ix].sum()),positive_rate=float(y.iloc[ix].mean()),auc=float(auc),log_loss=float(log_loss(y.iloc[ix],p.iloc[ix],labels=[0,1])))
            segments.append(row)
            lines.append(f"| {col} | {value} | {len(ix):,} | {row['positives']:,} | {row['positive_rate']:.3f} | {auc:.6f} | {row['log_loss']:.6f} |")
    (out/'segments.json').write_text(json.dumps(segments,indent=2)+'\n')
    lines += ['', 'Rural rows have lower AUC than Urban rows, and the Medium range-anxiety segment has lower AUC than Low. These are candidates for further error inspection, not evidence that a new feature will help. High range-anxiety has very few positives; its apparently strong AUC should not be treated as stable.', '', 'Segment differences are descriptive, depend on class prevalence and difficulty, and do not establish causal effects or fairness. Synthetic data can contain generator artifacts.', '',
              '![Calibration](calibration.png)', '',
              '## Reproducibility and limitations', '',
              '- Full configs, source/data/fold SHA-256 fingerprints, Git revision, and individual fold metrics are in [metrics](metrics/). Dependencies are pinned in requirements.txt.',
              '- Raw data, OOF rows, submission files and model binaries stay in ignored local artifacts. Download data through Kaggle under competition rules.',
              '- All models ran on CPU with eight threads each. No GPU, external data, copied competition notebook, pseudo-labeling or ensemble search was used.',
              '- Random stratified CV assumes exchangeable rows. This does not demonstrate temporal or customer-group generalization. Data checks report duplicate feature rows.',
              '- AUC measures ranking; probability calibration and practical thresholds require further independent validation.',
              '- Final private leaderboard results are unavailable while the competition is open. No final rank or medal is claimed.', '']
    (out/'ev-purchases.md').write_text('\n'.join(lines))
    names=sorted(runs, key=lambda n:runs[n]['auc_mean'])
    fig, ax=plt.subplots(figsize=(9,4))
    ax.errorbar([runs[n]['auc_mean'] for n in names], range(len(names)), xerr=[runs[n]['auc_std'] for n in names], fmt='o', capsize=4)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(min(r['auc_mean'] for r in runs.values())-.003,max(r['auc_mean'] for r in runs.values())+.002)
    ax.set_xlabel('Mean fold ROC-AUC ± one fold SD (not a confidence interval)')
    fig.tight_layout(); fig.savefig(out/'cv-comparison.png',dpi=160); plt.close(fig)
    frac, mean=calibration_curve(y,p,n_bins=10,strategy='quantile')
    fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],'--',color='gray'); ax.plot(mean,frac,'o-')
    ax.set(xlabel='Mean predicted probability',ylabel='Observed positive rate',title=f'OOF calibration: {winner}')
    fig.tight_layout(); fig.savefig(out/'calibration.png',dpi=160); plt.close(fig)
    print(json.dumps({'winner':winner,'mean_auc':runs[winner]['auc_mean'],'data':data_summary},indent=2))

if __name__ == '__main__':
    main()
