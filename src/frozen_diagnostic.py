"""Train once on folds 0–2; diagnose errors on disjoint, unseen folds 3 and 4."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .investigate import DATA,ORIG
from .offset_boost import OffsetEncoder,OffsetXGB
from .pipeline import digest,score


def main():
    df=pd.read_csv(DATA/'train.csv');orig=pd.read_csv(ORIG);folds=pd.read_csv('artifacts/r4-offset/oof.csv');assert df.id.equals(folds.id)
    train=folds.fold.le(2);heldout=~train;y=df.Will_Buy_EV.eq('Yes')
    out=Path('artifacts/r5-frozen-diagnostic');out.mkdir(exist_ok=True,parents=True)
    enc=OffsetEncoder('full');xt=enc.fit_transform(df[train],y[train],orig);xh=enc.transform(df[heldout])
    params=dict(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.9,colsample_bytree=.8,
                reg_lambda=10.,reg_alpha=1.,min_child_weight=5,max_bin=512,device='cuda',tree_method='hist',
                eval_metric='auc',n_jobs=8,random_state=42)
    model=OffsetXGB(**params);model.fit(xt,y[train])
    prediction=model.predict_proba(xh)[:,1]
    pd.DataFrame({'id':df.id[heldout],'target':y[heldout].astype(int),'fold':folds.fold[heldout],'prediction':prediction}).to_csv(out/'heldout.csv',index=False)
    joblib.dump({'encoder':enc,'model':model},out/'model.joblib')
    result={'training_folds':[0,1,2],'discovery_fold':3,'confirmation_fold':4,'params':params,
            'heldout_metrics':score(y[heldout],prediction),'source_sha256':digest(__file__),
            'caveat':'Fixed 600-tree budget informed by earlier development runs; this is a diagnostic split, not a new pristine final test.'}
    Path('reports/round5/frozen-model.json').write_text(json.dumps(result,indent=2)+'\n');print(result['heldout_metrics'])

if __name__=='__main__':main()
