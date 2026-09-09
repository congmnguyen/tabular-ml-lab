"""Paired repeated holdouts with fixed training budgets, no scored-fold stopping."""
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from . import expert
from .low_smoothing import encoder_factory
from .modulo_features import ModuloEncoder
from .joint_effects import JointEncoder
from .source_context import SourceContextEncoder
from .offset_boost import OffsetXGB
from .investigate import DATA, ORIG
from .pipeline import score, digest

SEEDS=[17,91,2026]
PARAMS=dict(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.9,colsample_bytree=.8,reg_lambda=10.,reg_alpha=1.,min_child_weight=5,max_bin=512,device='cuda',tree_method='hist',eval_metric='auc',n_jobs=8,random_state=42)


def main():
    expert.TargetEncoder=encoder_factory
    df=pd.read_csv(DATA/'train.csv');original=pd.read_csv(ORIG);y=df.Will_Buy_EV.eq('Yes').astype(int).to_numpy()
    out=Path('artifacts/r11-stability');out.mkdir(parents=True,exist_ok=True);results=[]
    for seed in SEEDS:
        tr,va=next(StratifiedShuffleSplit(n_splits=1,test_size=.1,random_state=seed).split(df,y))
        for name,cls in [('baseline',ModuloEncoder),('joint',JointEncoder),('source',SourceContextEncoder)]:
            prefix=out/f'{name}-{seed}'
            if prefix.with_suffix('.json').exists():
                results.append(json.loads(prefix.with_suffix('.json').read_text()));continue
            start=time.monotonic();encoder=cls('full');xt=encoder.fit_transform(df.iloc[tr],y[tr],original);xv=encoder.transform(df.iloc[va]);model=OffsetXGB(**PARAMS);model.fit(xt,y[tr])
            prediction=model.predict_proba(xv)[:,1]
            np.savez_compressed(prefix.with_suffix('.npz'),train_indices=tr,indices=va,prediction=prediction,target=y[va])
            joblib.dump({'encoder':encoder,'model':model},prefix.with_suffix('.joblib'))
            r={'name':name,'seed':seed,'train_rows':len(tr),'validation_rows':len(va),'metrics':score(y[va],prediction),'seconds':time.monotonic()-start,'model_params':PARAMS,'script_sha256':digest(__file__)}
            prefix.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);print(name,seed,r['metrics'],flush=True)
            del xt,xv,encoder,model;gc.collect()
    pairs=[]
    for seed in SEEDS:
        selected={r['name']:r for r in results if r['seed']==seed};base=selected['baseline']['metrics']['auc']
        pairs.append({'seed':seed,'baseline_auc':base,'joint_auc_gain':selected['joint']['metrics']['auc']-base,'source_auc_gain':selected['source']['metrics']['auc']-base})
    report={'results':results,'paired_differences':pairs,'mean_joint_gain':float(np.mean([r['joint_auc_gain'] for r in pairs])),'mean_source_gain':float(np.mean([r['source_auc_gain'] for r in pairs])),'caveat':'Same rows within each comparison, fixed 600 trees, no early stopping. Overlapping repeated holdouts and previously used data are development robustness checks, not independent final-test replicates. No Kaggle submission from partial holdouts.'}
    Path('reports/round11/stability.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',pairs,flush=True)

if __name__=='__main__':main()
