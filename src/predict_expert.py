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
    predictions=np.zeros(len(test),dtype=np.float64)
    summary=json.loads((Path(path)/'summary.json').read_text())
    splits=summary.get('args',{}).get('splits',5)
    for f in range(splits):
        checkpoint_path=Path(path)/f'fold{f}.pt'
        if checkpoint_path.exists():
            import torch
            from .neural import make_model, predict
            torch.set_num_threads(4)
            torch.set_float32_matmul_precision('high')
            checkpoint=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
            model=make_model(checkpoint['n_features'],checkpoint['k'],checkpoint['width'],checkpoint.get('bins')).cuda()
            model.load_state_dict(checkpoint['state'])
            encoder=joblib.load(Path(path)/f'encoder{f}.joblib')
            pp=predict(model,encoder.transform(test))
            del model,encoder
        else:
            b=joblib.load(Path(path)/f'fold{f}.joblib')
            x=b['encoder'].transform(test)
            pp=b['model'].predict_proba(x)[:,1]
        predictions += pp / splits
    return predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--test',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();test=pd.read_csv(a.test);path=Path(a.run)
    if (path/'blend.json').exists():
        info=json.loads((path/'blend.json').read_text());pred=np.zeros(len(test))
        for run,w in zip(info['runs'],info['selected']['weights']):
            if not w:continue
            pp=predict_run(run,test)
            if info['selected']['kind']=='rank':
                pred+=w*rankdata(pp)/len(pp)
            else:
                pred+=w*pp
    else:pred=predict_run(path,test)
    sub=pd.DataFrame({'id':test.id,'Will_Buy_EV':pred})
    validate_submission(sub,test,{'id':'id','target':'Will_Buy_EV'});sub.to_csv(a.output,index=False)
    print('Reproduced',len(sub),'rows',flush=True)

if __name__=='__main__':main()
