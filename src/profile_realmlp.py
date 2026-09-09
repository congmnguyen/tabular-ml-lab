"""Small training-only compatibility and resource profile for official RealMLP."""
import json,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from .realmlp_features import RealMLPFeatures
from .investigate import DATA


def main():
    df=pd.read_csv(DATA/'train.csv');outer=np.load('artifacts/r11-stability/baseline-17.npz')['train_indices']
    idx=np.random.default_rng(104).choice(outer,size=20000,replace=False);sample=df.iloc[idx];y=sample.Will_Buy_EV.eq('Yes').astype(int).to_numpy();results=[]
    for identity in [False,True]:
        features=RealMLPFeatures(identity);x=features.fit_transform(sample)
        model=RealMLP_TD_Classifier(device='cuda',random_state=42,n_cv=1,n_refit=0,n_ens=1,val_fraction=0.,n_epochs=2,stop_epoch=2,batch_size=1024,predict_batch_size=4096,use_ls=False,val_metric_name='cross_entropy',n_threads=4,verbosity=2)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();model.fit(x,y,cat_col_names=features.cat_columns)
        elapsed=time.monotonic()-start
        unseen=sample.iloc[:2].copy();unseen['Annual_Income_USD']=99999999.;unseen['Daily_Commute_km']=99999.;prediction=model.predict_proba(features.transform(unseen))
        assert np.isfinite(prediction).all()
        r={'identities':identity,'rows':len(x),'epochs':2,'seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'unseen_prediction_finite':True,'params':model.get_params()};results.append(r);print('PROFILE',identity,elapsed,r['peak_allocated_bytes'],flush=True)
        del model,x,features;gc.collect();torch.cuda.empty_cache()
    Path('reports/round13/profile.json').write_text(json.dumps(results,indent=2,default=str)+'\n')

if __name__=='__main__':main()
