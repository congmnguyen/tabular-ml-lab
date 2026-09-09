"""Compare full-statistic inference with a fixed five-subset prediction average."""
import gc
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from .investigate import DATA
from .pipeline import score,digest
from .stability_audit import SEEDS


def main():
    df=pd.read_csv(DATA/'train.csv');y=df.Will_Buy_EV.eq('Yes').astype(int).to_numpy();out=Path('artifacts/r11-stability');results=[]
    for seed in SEEDS:
        saved=np.load(out/f'baseline-{seed}.npz');tr=saved['train_indices'];va=saved['indices']
        assert not np.intersect1d(tr,va).size
        assert np.array_equal(y[va],saved['target'])
        bundle=joblib.load(out/f'baseline-{seed}.joblib');encoder=bundle['encoder'];model=bundle['model']
        train_keys=encoder.key_frame(encoder.prepare(df.iloc[tr]));valid_keys=encoder.key_frame(encoder.prepare(df.iloc[va]))
        xv=encoder.transform(df.iloc[va]);baseline=saved['prediction'];prediction=np.zeros(len(va));changes=np.zeros((len(va),3))
        monitored=['Annual_Income_USD_te0','Daily_Commute_km_te0','income_mod1000_te0'];baseline_values=xv[monitored].to_numpy(copy=True)
        for f,(inner_tr,_) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(train_keys,y[tr])):
            altered=xv.copy()
            for j,original in enumerate(encoder.encoders):
                te=TargetEncoder(target_type='binary',smooth=original.smooth)
                te.fit(train_keys.iloc[inner_tr],y[tr][inner_tr]);encoded=te.transform(valid_keys).astype(np.float32)
                for k,c in enumerate(encoder.keys):altered[c+f'_te{j}']=encoded[:,k]
                del te,encoded
            prediction+=model.predict_proba(altered)[:,1]/5
            changes+=np.abs(altered[monitored].to_numpy()-baseline_values)/5
            del altered;gc.collect()
        counts=df.iloc[va].Annual_Income_USD.map(df.iloc[tr].Annual_Income_USD.value_counts()).fillna(0).to_numpy()
        groups=[]
        for label,low,high in [('unseen',0,1),('1-4',1,5),('5-19',5,20),('20-99',20,100),('100+',100,np.inf)]:
            mask=(counts>=low)&(counts<high)
            if mask.any():groups.append({'income_count':label,'rows':int(mask.sum()),'mean_absolute_encoding_change':dict(zip(monitored,changes[mask].mean(axis=0).tolist())),'mean_absolute_prediction_change':float(np.abs(prediction[mask]-baseline[mask]).mean())})
        before=score(y[va],baseline);after=score(y[va],prediction)
        r={'seed':seed,'baseline':before,'matched_support_average':after,'auc_gain':after['auc']-before['auc'],'frequency_groups':groups};results.append(r)
        np.savez_compressed(out/f'matched-{seed}.npz',indices=va,target=y[va],prediction=prediction)
        (out/f'matched-{seed}.json').write_text(json.dumps(r,indent=2)+'\n');print(seed,r['auc_gain'],flush=True)
        del train_keys,valid_keys,xv,encoder,model,bundle;gc.collect()
    gains=[r['auc_gain'] for r in results];report={'results':results,'mean_auc_gain':float(np.mean(gains)),'passes_expansion_gate':bool(min(gains)>0 and np.mean(gains)>.00002),'script_sha256':digest(__file__),'protocol':'Five fixed 80%-support inner encoders, average model probabilities. No validation labels enter fit. Models unchanged; frequency/source/non-TE features unchanged. Overlapping development holdouts; not independent final tests.'}
    Path('reports/round11/encoding-support.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',report['mean_auc_gain'],report['passes_expansion_gate'],flush=True)

if __name__=='__main__':main()
