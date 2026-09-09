"""Conditional permutation diagnostics on a saved out-of-fold XGBoost model."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .investigate import DATA


def main():
    df=pd.read_csv(DATA/'train.csv');saved=np.load('artifacts/r2-full-xgb/fold0.npz');df=df.iloc[saved['indices']].reset_index(drop=True)
    y=df.Will_Buy_EV.eq('Yes').to_numpy();bundle=joblib.load('artifacts/r2-full-xgb/fold0.joblib')
    base=float(roc_auc_score(y,saved['oof']));rng=np.random.default_rng(42)
    core=df[['Environmental_Concern_Level','Subsidy_Available','Range_Anxiety_Level']].astype(str).agg('|'.join,axis=1)
    strata=core+'|'+np.floor(df.Annual_Income_USD/1000).astype(str)
    blocks={
        'income_identity_within_1k_and_core':['Annual_Income_USD'],
        'commute_within_1k_and_core':['Daily_Commute_km'],
        'other_covariates_within_1k_and_core':['Age','Gender','City_Type','Current_Car_Type','Number_of_Cars_Owned','Home_Charging_Possible','Charging_Stations_Near_Home','Charging_Stations_Near_Work'],
    }
    rows=[]
    groups=list(pd.Series(np.arange(len(df))).groupby(strata).indices.values())
    for name,columns in blocks.items():
        for seed in [42,43]:
            rng=np.random.default_rng(seed);permutation=np.arange(len(df))
            for indices in groups:permutation[indices]=rng.permutation(indices)
            altered=df.copy()
            for col in columns:altered[col]=df.iloc[permutation][col].to_numpy()
            pred=bundle['model'].predict_proba(bundle['encoder'].transform(altered))[:,1]
            auc=float(roc_auc_score(y,pred));row={'block':name,'seed':seed,'auc':auc,'auc_drop':base-auc,'rows_moved':int((permutation!=np.arange(len(df))).sum())}
            rows.append(row);print(row,flush=True)
    result={'fold':0,'model':'r2-full-xgb','baseline_auc':base,'strata':'income floor/1000 × concern × subsidy × anxiety',
            'warning':'Conditional permutation is an approximate diagnostic, not a causal effect or an independent model-selection holdout.', 'results':rows}
    Path('reports/round4/conditional-permutation.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
