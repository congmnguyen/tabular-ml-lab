"""Independently check public expert hypotheses against available data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score, log_loss

DATA=Path.home()/'.cache/kaggle/playground-series-s6e9'
ORIG=Path.home()/'.cache/kaggle/ev-original/EV_Adoption_and_Range_Anxiety_Dataset.csv'

def recipe(df):
    return (1.2*df.Annual_Income_USD/100000 + .6*df.Environmental_Concern_Level
            +2*df.Subsidy_Available.eq('Yes')-df.Range_Anxiety_Level.eq('Medium').astype(float)
            -3*df.Range_Anxiety_Level.eq('High'))

def main():
    train,test,orig=pd.read_csv(DATA/'train.csv'),pd.read_csv(DATA/'test.csv'),pd.read_csv(ORIG)
    result={}
    for name,df in [('original',orig),('competition',train)]:
        y=df.Will_Buy_EV.eq('Yes').astype(int)
        valid=df[['Annual_Income_USD','Environmental_Concern_Level']].notna().all(axis=1)
        sc=recipe(df.loc[valid])
        result[name]={'rows':len(df),'missing':df.isna().sum().to_dict(),'recipe_auc':roc_auc_score(y[valid],sc),
                      'recipe_logloss':log_loss(y[valid],np.clip(ndtr(sc-5.5),1e-8,1-1e-8)),
                      'income_30k_count':int(df.Annual_Income_USD.eq(30000).sum()),
                      'income_unique':int(df.Annual_Income_USD.nunique()),'positive_rate':float(y.mean())}
    tables=[]
    for name,df in [('original',orig),('competition',train)]:
        df=df.assign(y=df.Will_Buy_EV.eq('Yes').astype(int))
        for label,mask in [('0–2',df.Charging_Stations_Near_Home.between(0,2)),('11–14',df.Charging_Stations_Near_Home.between(11,14))]:
            for home in ['All','Yes','No']:
                subset=df[mask] if home=='All' else df[mask & df.Home_Charging_Possible.eq(home)]
                tables.append(dict(dataset=name,stations=label,home=home,rows=len(subset),rate=float(subset.y.mean())))
    result['charging_strata']=tables
    base=pd.read_csv('artifacts/lightgbm-features/oof.csv')
    residual=base.target-base.prediction
    digit=(train.Annual_Income_USD.astype(int)%10)
    digit_stats=pd.DataFrame({'digit':digit,'residual':residual,'y':base.target}).groupby('digit').agg(rows=('y','size'),rate=('y','mean'),mean_residual=('residual','mean'))
    result['income_last_digit_residuals']=digit_stats.reset_index().to_dict('records')
    feature_cols=[c for c in train if c not in ['id','Will_Buy_EV']]
    orig_hash=pd.util.hash_pandas_object(orig[feature_cols],index=False)
    result['exact_original_feature_matches']={name:int(pd.util.hash_pandas_object(df[feature_cols],index=False).isin(orig_hash).sum()) for name,df in [('train',train),('test',test)]}
    result['income_cliff']={}
    for label,mask in [('income>=170537',train.Annual_Income_USD.ge(170537)),('income38000to42000',train.Annual_Income_USD.between(38000,42000))]:
        result['income_cliff'][label]={'rows':int(mask.sum()),'rate':float(base.target[mask].mean())}
    out=Path('reports/round2');out.mkdir(exist_ok=True)
    (out/'data-investigation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
