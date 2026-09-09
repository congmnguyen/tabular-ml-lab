"""Assemble completed fold artifacts with split and coverage verification."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .pipeline import score, digest, validate_submission


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    run=Path(a.run);root=Path.home()/'.cache/kaggle/playground-series-s6e9'
    df=pd.read_csv(root/'train.csv');test=pd.read_csv(root/'test.csv');y=df.Will_Buy_EV.eq('Yes').to_numpy()
    oof=np.full(len(df),np.nan);pred=np.zeros(len(test),dtype=np.float64);fold_id=np.full(len(df),-1)
    metrics=[]
    for fold,(_,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y)):
        saved=np.load(run/f'fold{fold}.npz')
        np.testing.assert_array_equal(saved['indices'],valid)
        assert saved['oof'].shape==(len(valid),) and saved['test'].shape==(len(test),)
        oof[valid]=saved['oof'];fold_id[valid]=fold;pred+=saved['test']/5
        metrics.append(json.loads((run/f'fold{fold}.json').read_text()))
    assert np.isfinite(oof).all() and ((oof>=0)&(oof<=1)).all()
    pd.DataFrame({'id':df.id,'target':y.astype(int),'fold':fold_id,'prediction':oof}).to_csv(run/'oof.csv',index=False)
    sub=pd.DataFrame({'id':test.id,'Will_Buy_EV':pred});validate_submission(sub,test,{'id':'id','target':'Will_Buy_EV'})
    sub.to_csv(run/'submission.csv',index=False)
    result={'run':str(run),'oof':score(y,oof),'mean_auc':float(np.mean([m['auc'] for m in metrics])),
            'folds':metrics,'train_sha256':digest(root/'train.csv'),'test_sha256':digest(root/'test.csv'),
            'early_stopping_on_scored_fold':True}
    (run/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result['oof'])

if __name__=='__main__':main()
