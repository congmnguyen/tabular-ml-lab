"""Controlled official RealMLP raw versus identity-embedding holdout comparison."""
import gc,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .realmlp_features import RealMLPFeatures
from .investigate import DATA
from .stability_audit import SEEDS
from .pipeline import score,digest

PARAMS=dict(device='cuda',random_state=42,n_cv=1,n_refit=0,n_ens=1,val_fraction=0.,n_epochs=64,stop_epoch=64,batch_size=1024,predict_batch_size=4096,use_ls=False,val_metric_name='cross_entropy',n_threads=4,verbosity=2)


def main():
    import torch
    from pytabkit import RealMLP_TD_Classifier
    df=pd.read_csv(DATA/'train.csv');y=df.Will_Buy_EV.eq('Yes').astype(int).to_numpy()
    out=Path('artifacts/r13-realmlp');out.mkdir(parents=True,exist_ok=True);results=[]
    for seed in SEEDS:
        reference=np.load(f'artifacts/r11-stability/baseline-{seed}.npz');tr=reference['train_indices'];va=reference['indices'];baseline=reference['prediction'];baseline_metrics=score(y[va],baseline)
        for identities in [False,True]:
            name='identities' if identities else 'continuous';prefix=out/f'{name}-{seed}'
            if prefix.with_suffix('.json').exists():results.append(json.loads(prefix.with_suffix('.json').read_text()));continue
            features=RealMLPFeatures(identities);xt=features.fit_transform(df.iloc[tr]);xv=features.transform(df.iloc[va]);model=RealMLP_TD_Classifier(**PARAMS)
            start=time.monotonic();torch.cuda.reset_peak_memory_stats();print('START',name,seed,len(tr),flush=True)
            model.fit(xt,y[tr],cat_col_names=features.cat_columns)
            prediction=model.predict_proba(xv)[:,1];assert np.isfinite(prediction).all()
            blends=[]
            for w in [0.,.1,.2]:
                p=(1-w)*baseline+w*prediction;metrics=score(y[va],p)
                blends.append({'neural_weight':w,'metrics':metrics,'auc_gain':metrics['auc']-baseline_metrics['auc']})
            r={'name':name,'seed':seed,'metrics':score(y[va],prediction),'baseline_metrics':baseline_metrics,'blends':blends,'prediction_correlation':float(np.corrcoef(baseline,prediction)[0,1]),'seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'params':model.get_params(),'categorical_cardinalities':{c:len(v) for c,v in features.vocabularies.items()},'script_sha256':digest(__file__),'feature_script_sha256':digest('src/realmlp_features.py')}
            np.savez_compressed(prefix.with_suffix('.npz'),indices=va,target=y[va],prediction=prediction)
            model.to('cpu');joblib.dump({'features':features,'model':model},prefix.with_suffix('.joblib'))
            prefix.with_suffix('.json').write_text(json.dumps(r,indent=2,default=str)+'\n');results.append(r)
            print('RESULT',name,seed,r['metrics'],'blend gains',[b['auc_gain'] for b in blends],flush=True)
            del features,xt,xv,model;gc.collect();torch.cuda.empty_cache()
    summary=[]
    for name in ['continuous','identities']:
        for weight in [.1,.2]:
            gains=[next(b['auc_gain'] for b in r['blends'] if b['neural_weight']==weight) for r in results if r['name']==name]
            summary.append({'name':name,'neural_weight':weight,'gains':gains,'mean_auc_gain':float(np.mean(gains)),'passes_gate':bool(min(gains)>0 and np.mean(gains)>.00002)})
    report={'results':results,'summary':summary,'protocol':'Fixed 64 epochs; all outer training rows used; no scored-fold or internal best-epoch selection. Shared numeric features; categorical identity view is the only feature difference. Reused overlapping development holdouts.'}
    Path('reports/round13/realmlp.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print('COMPLETE',summary,flush=True)

if __name__=='__main__':main()
