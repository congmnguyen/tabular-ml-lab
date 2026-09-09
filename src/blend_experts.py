"""Compare a small predeclared grid of blends using aligned OOF predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from .pipeline import score


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--small-third',action='store_true',help='Screen a complementary third model at 0/5/10/20 percent')
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    oofs=[pd.read_csv(Path(r)/'oof.csv') for r in a.runs]
    tests=[pd.read_csv(Path(r)/'submission.csv') for r in a.runs]
    first=oofs[0]
    for o,t in zip(oofs,tests):
        assert np.array_equal(o[['id','target','fold']].to_numpy(),first[['id','target','fold']].to_numpy())
        assert t.id.equals(tests[0].id)
    y=first.target.to_numpy();fold=first.fold.to_numpy()
    configs=[]
    for i in range(len(a.runs)):
        w=np.zeros(len(a.runs));w[i]=1;configs.append(('single',w))
    if len(a.runs)==2:
        for w in [.25,.5,.75]:configs.append(('probability',np.array([w,1-w])))
        configs.append(('rank',np.array([.5,.5])))
    else:
        configs.append(('probability',np.ones(len(a.runs))/len(a.runs)))
    if a.small_third:
        if len(a.runs)!=3:raise ValueError('--small-third requires exactly three runs')
        configs=[(kind,np.array([(1-w)/2,(1-w)/2,w])) for kind in ['probability','rank'] for w in [0.,.05,.1,.2]]
    results=[];best=None
    for kind,weights in configs:
        pred=np.zeros(len(y));test_pred=np.zeros(len(tests[0]))
        for w,o,t in zip(weights,oofs,tests):
            if kind=='rank':
                ranked=np.zeros(len(y))
                for f in np.unique(fold):ranked[fold==f]=rankdata(o.prediction[fold==f])/sum(fold==f)
                pred+=w*ranked;test_pred+=w*rankdata(t.Will_Buy_EV)/len(t)
            else:pred+=w*o.prediction.to_numpy();test_pred+=w*t.Will_Buy_EV.to_numpy()
        aucs=[roc_auc_score(y[fold==f],pred[fold==f]) for f in np.unique(fold)]
        result={'kind':kind,'weights':weights.tolist(),'mean_auc':float(np.mean(aucs)),'fold_auc':aucs,'oof_auc':float(roc_auc_score(y,pred))}
        results.append(result)
        if best is None or result['mean_auc']>best[0]['mean_auc']:best=(result,pred,test_pred)
    selected,pred,test_pred=best
    pd.DataFrame({'id':first.id,'target':y,'fold':fold,'prediction':pred}).to_csv(out/'oof.csv',index=False)
    pd.DataFrame({'id':tests[0].id,'Will_Buy_EV':test_pred}).to_csv(out/'submission.csv',index=False)
    info={'runs':a.runs,'candidates':results,'selected':selected,'selection':'mean fold AUC; fixed weight grid',
          'prediction_correlations':np.corrcoef([o.prediction for o in oofs]).tolist()}
    (out/'blend.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
