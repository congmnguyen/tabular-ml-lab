"""Compare aligned development-fold predictions; never use leaderboard labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .pipeline import digest


def main():
    y=pd.read_csv(Path.home()/'.cache/kaggle/playground-series-s6e9/train.csv',usecols=['Will_Buy_EV']).Will_Buy_EV.eq('Yes').to_numpy()
    left=np.load('artifacts/r2-full-lgb/fold0.npz'); right=np.load('artifacts/r2-full-xgb/fold0.npz')
    np.testing.assert_array_equal(left['indices'],right['indices'])
    target=y[left['indices']]; base=(left['oof']+right['oof'])/2
    baseline=float(roc_auc_score(target,base)); rows=[]
    for path in sorted(Path('artifacts').glob('r3-*/fold0.npz')):
        run=path.parent
        if not (run/'fold0.json').exists():continue
        p=np.load(path);np.testing.assert_array_equal(left['indices'],p['indices'])
        assert np.isfinite(p['oof']).all() and ((p['oof']>=0)&(p['oof']<=1)).all()
        blends=[{'candidate_weight':w,'auc':float(roc_auc_score(target,(1-w)*base+w*p['oof']))} for w in [.05,.1,.25,.5,1.]]
        row={'run':run.name,'metrics':json.loads((run/'fold0.json').read_text()),
             'prediction_sha256':digest(path),'correlation':float(np.corrcoef(base,p['oof'])[0,1]),'blends':blends}
        rows.append(row)
        print(run.name, 'single',f"{row['metrics']['auc']:.9f}",'best blend',max(blends,key=lambda v:v['auc']))
    report={'fold':0,'baseline_auc':baseline,'baseline_runs':['r2-full-lgb','r2-full-xgb'],
            'warning':'Repeated development-fold screening and early stopping create selection optimism. Full OOF confirmation required.', 'screens':rows}
    Path('reports/round3/screens.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
