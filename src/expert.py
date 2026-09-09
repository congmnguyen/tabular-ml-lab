"""Round-two EV features and boosting. Research attribution: reports/round2/research.md."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from .pipeline import score, digest, validate_submission
from .investigate import recipe

TARGET='Will_Buy_EV'


def raw_features(df, mode):
    x=df.drop(columns=['id','Buyer_ID',TARGET],errors='ignore').copy()
    x['recipe']=recipe(df)
    x['worry']=df.Daily_Commute_km-5*(df.Charging_Stations_Near_Home+df.Charging_Stations_Near_Work)-150*df.Home_Charging_Possible.eq('Yes')
    x['subsidy_income']=df.Annual_Income_USD/1e5*df.Subsidy_Available.eq('Yes')
    x['subsidy_concern']=df.Environmental_Concern_Level*df.Subsidy_Available.eq('Yes')
    if mode=='recipe':return x
    nums=df.drop(columns=['id','Buyer_ID',TARGET],errors='ignore').select_dtypes(include='number').columns
    for col in nums:
        values=df[col].fillna(0)
        for power in range(-4,4):
            # Matches the public floating-point digit hypothesis, including rounding artifacts.
            x[f'{col}_digit_{power}']=(values // (10.**power) % 10).astype('int8')
    x['income_bin100']=np.floor(df.Annual_Income_USD/100)
    x['income_bin1000']=np.floor(df.Annual_Income_USD/1000)
    x['commute_bin1']=np.floor(df.Daily_Commute_km)
    return x


class ExpertEncoder:
    def __init__(self,mode='full',seed=42):self.mode,self.seed=mode,seed

    def prepare(self,df):
        x=raw_features(df,self.mode)
        return x

    def fit_transform(self,df,y,original):
        x=self.prepare(df)
        self.columns=list(x)
        self.numeric=x.select_dtypes(include='number').columns.tolist()
        self.categories=[c for c in x if c not in self.numeric]
        self.codes={c:{v:i for i,v in enumerate(sorted(x[c].fillna('__NA__').astype(str).unique()))} for c in self.categories}
        self.freq={c:x[c].value_counts(normalize=True).to_dict() for c in x}
        self.orig={}
        self.orig_mean=float(original[TARGET].eq('Yes').mean())
        if self.mode=='full':
            for c in original:
                if c in df and c not in ['Buyer_ID','id',TARGET]:
                    self.orig[c]=original.assign(_y=original[TARGET].eq('Yes').astype(int)).groupby(c,observed=True)._y.mean().to_dict()
        # Drop constant keys according to outer training fold only.
        self.keys=[c for c in x if x[c].nunique()>1] if self.mode!='recipe' else []
        self.encoders=[]
        enc=[]
        for smooth in ([10.,100.] if self.mode=='full' else []):
            te=TargetEncoder(target_type='binary',smooth=smooth,cv=5,shuffle=True,random_state=self.seed)
            enc.append(te.fit_transform(self.key_frame(x),y).astype(np.float32))
            self.encoders.append(te)
        return self.assemble(df,x,enc)

    def key_frame(self,x):
        return x[self.keys].fillna('__NA__').astype(str)

    def assemble(self,df,x,enc):
        cols={}
        for c in self.numeric:cols[c]=x[c].to_numpy(dtype=np.float32)
        for c in self.categories:cols[c]=x[c].fillna('__NA__').astype(str).map(self.codes[c]).fillna(-1).to_numpy(dtype=np.float32)
        if self.mode!='recipe':
            for c in self.columns:cols[c+'_frequency']=x[c].map(self.freq[c]).fillna(0).to_numpy(dtype=np.float32)
        for c,mapping in self.orig.items():cols[c+'_original_mean']=df[c].map(mapping).fillna(self.orig_mean).to_numpy(dtype=np.float32)
        for i,mat in enumerate(enc):
            for j,c in enumerate(self.keys):cols[f'{c}_te{i}']=mat[:,j]
        return pd.DataFrame(cols,index=df.index).astype(np.float32)

    def transform(self,df):
        x=self.prepare(df)
        return self.assemble(df,x,[e.transform(self.key_frame(x)).astype(np.float32) for e in self.encoders])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(Path.home()/'.cache/kaggle/playground-series-s6e9'))
    p.add_argument('--original',default=str(Path.home()/'.cache/kaggle/ev-original/EV_Adoption_and_Range_Anxiety_Dataset.csv'))
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['recipe','digits','full'],default='full')
    p.add_argument('--model',choices=['lgb','xgb'],default='lgb')
    p.add_argument('--folds',default='0,1,2,3,4')
    p.add_argument('--rounds',type=int,default=5000)
    p.add_argument('--depth',type=int,default=5)
    p.add_argument('--rate',type=float,default=.03)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(Path(a.data)/'train.csv');test=pd.read_csv(Path(a.data)/'test.csv');orig=pd.read_csv(a.original)
    y=df[TARGET].eq('Yes').astype(int)
    splitter=list(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y))
    todo=[int(f) for f in a.folds.split(',')]
    for fold in todo:
        if (out/f'fold{fold}.json').exists():continue
        tick=time.monotonic();tr,va=splitter[fold]
        enc=ExpertEncoder(a.mode)
        xt=enc.fit_transform(df.iloc[tr],y.iloc[tr],orig)
        xv=enc.transform(df.iloc[va]);xx=enc.transform(test)
        print(f'fold={fold} encoding seconds={time.monotonic()-tick:.1f} shape={xt.shape}',flush=True)
        if a.model=='lgb':
            params=dict(n_estimators=a.rounds,learning_rate=a.rate,max_depth=a.depth,num_leaves=2**a.depth,
                        min_child_samples=30,colsample_bytree=.5,reg_alpha=.07,reg_lambda=3.,max_bin=255,
                        n_jobs=8,verbosity=-1,random_state=42,deterministic=True,force_col_wise=True)
            model=LGBMClassifier(**params)
            model.fit(xt,y.iloc[tr],eval_set=[(xv,y.iloc[va])],eval_metric='auc',callbacks=[early_stopping(300,first_metric_only=True,verbose=False),log_evaluation(500)])
            iteration=int(model.best_iteration_)
        else:
            params=dict(n_estimators=a.rounds,learning_rate=a.rate,max_depth=a.depth,subsample=.9,colsample_bytree=.8,
                        reg_lambda=10.,reg_alpha=1.,min_child_weight=5,max_bin=512,device='cuda',tree_method='hist',
                        eval_metric='auc',early_stopping_rounds=250,n_jobs=8,random_state=42)
            model=XGBClassifier(**params)
            model.fit(xt,y.iloc[tr],eval_set=[(xv,y.iloc[va])],verbose=500)
            iteration=int(model.best_iteration+1)
        pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xx)[:,1]
        np.savez_compressed(out/f'fold{fold}.npz',indices=va,oof=pv,test=pt)
        # Store preprocessing with estimator for independent inference.
        joblib.dump(dict(encoder=enc,model=model),out/f'fold{fold}.joblib')
        result={'fold':fold,**score(y.iloc[va],pv),'best_iteration':iteration,'seconds':time.monotonic()-tick,
                'params':params,'features':list(xt),'source_sha256':digest(__file__)}
        (out/f'fold{fold}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['params','features']}),flush=True)
        del xt,xv,xx,model,enc
    files=[out/f'fold{f}.npz' for f in range(5)]
    if all(f.exists() for f in files):
        oof=np.full(len(df),np.nan);pred=np.zeros(len(test));folds=np.full(len(df),-1)
        results=[]
        for i,f in enumerate(files):
            z=np.load(f);oof[z['indices']]=z['oof'];pred+=z['test']/5;folds[z['indices']]=i
            results.append(json.loads((out/f'fold{i}.json').read_text()))
        assert np.isfinite(oof).all()
        pd.DataFrame({'id':df.id,'target':y,'fold':folds,'prediction':oof}).to_csv(out/'oof.csv',index=False)
        sub=pd.DataFrame({'id':test.id,TARGET:pred});validate_submission(sub,test,{'id':'id','target':TARGET});sub.to_csv(out/'submission.csv',index=False)
        summary={'args':vars(a),'oof':score(y,oof),'mean_auc':float(np.mean([r['auc'] for r in results])),
                 'folds':results,'train_sha256':digest(Path(a.data)/'train.csv'),'original_sha256':digest(a.original),
                 'early_stopping_on_scored_fold':True}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',summary['oof'],flush=True)

if __name__=='__main__':
    from src.expert import main as entry
    entry()
