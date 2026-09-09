"""Reproduce an expert run or selected blend from trusted saved fold models."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .pipeline import validate_submission


def predict_run(path,test):
    predictions=[]
    for f in range(5):
        b=joblib.load(Path(path)/f'fold{f}.joblib')
        x=b['encoder'].transform(test)
        predictions.append(b['model'].predict_proba(x)[:,1])
    return np.mean(predictions,axis=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--test',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();test=pd.read_csv(a.test);path=Path(a.run)
    if (path/'blend.json').exists():
        info=json.loads((path/'blend.json').read_text());pred=np.zeros(len(test))
        for run,w in zip(info['runs'],info['selected']['weights']):
            if not w:continue
            pp=predict_run(run,test)
            if info['selected']['kind']=='rank':pp=rankdata(pp)/len(pp)
            pred+=w*pp
    else:pred=predict_run(path,test)
    sub=pd.DataFrame({'id':test.id,'Will_Buy_EV':pred})
    validate_submission(sub,test,{'id':'id','target':'Will_Buy_EV'});sub.to_csv(a.output,index=False)
    print('Reproduced',len(sub),'rows',flush=True)

if __name__=='__main__':main()
