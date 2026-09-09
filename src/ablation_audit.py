"""Predeclared feature-family removals on matched fixed-budget holdouts."""
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from . import expert
from .low_smoothing import encoder_factory
from .modulo_features import ModuloEncoder
from .offset_boost import OffsetXGB
from .investigate import DATA,ORIG
from .stability_audit import SEEDS,PARAMS
from .pipeline import score,digest


def dropped_columns(columns,group):
    rules={'fractional_digits':lambda c:'_digit_-' in c,'source_means':lambda c:c.endswith('_original_mean'),'frequency':lambda c:c.endswith('_frequency'),'low_smoothing':lambda c:c.endswith('_te0'),'high_smoothing':lambda c:c.endswith('_te1')}
    return [c for c in columns if rules[group](c)]


def main():
    expert.TargetEncoder=encoder_factory
    df=pd.read_csv(DATA/'train.csv');original=pd.read_csv(ORIG);y=df.Will_Buy_EV.eq('Yes').astype(int).to_numpy()
    out=Path('artifacts/r12-ablation');out.mkdir(parents=True,exist_ok=True);results=[]
    groups=['fractional_digits','source_means','frequency','low_smoothing','high_smoothing']
    for seed in SEEDS:
        saved=np.load(f'artifacts/r11-stability/baseline-{seed}.npz');tr=saved['train_indices'];va=saved['indices']
        enc=ModuloEncoder('full');xt=enc.fit_transform(df.iloc[tr],y[tr],original);xv=enc.transform(df.iloc[va])
        joblib.dump(enc,out/f'encoder-{seed}.joblib')
        baseline=score(y[va],saved['prediction'])
        for group in groups:
            prefix=out/f'{group}-{seed}'
            if prefix.with_suffix('.json').exists():results.append(json.loads(prefix.with_suffix('.json').read_text()));continue
            tick=time.monotonic();drop=dropped_columns(list(xt),group);assert drop,group
            model=OffsetXGB(**PARAMS);model.fit(xt.drop(columns=drop),y[tr]);pred=model.predict_proba(xv.drop(columns=drop))[:,1]
            np.savez_compressed(prefix.with_suffix('.npz'),indices=va,target=y[va],prediction=pred)
            joblib.dump({'model':model,'drop_columns':drop,'encoder_path':str(out/f'encoder-{seed}.joblib')},prefix.with_suffix('.joblib'))
            metrics=score(y[va],pred);r={'group':group,'seed':seed,'dropped_columns':drop,'metrics':metrics,'baseline_metrics':baseline,'auc_gain':metrics['auc']-baseline['auc'],'seconds':time.monotonic()-tick,'script_sha256':digest(__file__)}
            prefix.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);print(group,seed,r['auc_gain'],flush=True)
            del model;gc.collect()
        del xt,xv,enc;gc.collect()
    summary=[]
    for group in groups:
        gains=[r['auc_gain'] for r in results if r['group']==group];summary.append({'group':group,'gains':gains,'mean_auc_gain':float(np.mean(gains)),'passes_gate':bool(min(gains)>0 and np.mean(gains)>.00002)})
    report={'results':results,'summary':summary,'caveat':'Five predeclared removals on three reused, overlapping development holdouts. Baseline and candidates use same rows and 600 trees, no early stopping. Multiple comparisons remain optimistic; no partial-holdout submission.'}
    Path('reports/round12/ablation.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',summary,flush=True)

if __name__=='__main__':main()
