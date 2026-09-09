"""Reproduce saved TabM fold predictions without training."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .neural import make_model, predict


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--input',default=str(Path.home()/'.cache/kaggle/playground-series-s6e9/test.csv'))
    p.add_argument('--output',required=True)
    a=p.parse_args(); torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    run=Path(a.run); frame=pd.read_csv(a.input)
    checkpoint=torch.load(run/f'fold{a.fold}.pt',weights_only=True,map_location='cpu')
    model=make_model(checkpoint['n_features'],checkpoint['k'],checkpoint['width'],checkpoint.get('bins')).cuda()
    model.load_state_dict(checkpoint['state'])
    encoder=joblib.load(run/f'encoder{a.fold}.joblib')
    prediction=predict(model,encoder.transform(frame))
    assert np.isfinite(prediction).all()
    pd.DataFrame({'id':frame.id,'Will_Buy_EV':prediction}).to_csv(a.output,index=False)
    print(json.dumps({'rows':len(frame),'output':a.output}))

if __name__=='__main__':main()
