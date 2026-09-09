"""Jointly estimate regularized exact-value effects and smooth numeric responses."""
import argparse,json,time,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,SplineTransformer,StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .pipeline import score,digest,validate_submission

TARGET='Will_Buy_EV'

class ValueEncoder:
    def keys(self,df):
        x=df.drop(columns=['id',TARGET],errors='ignore').copy()
        for c in x:x[c]=x[c].fillna('__NA__').astype(str)
        for a,b in [('Environmental_Concern_Level','Subsidy_Available'),('Environmental_Concern_Level','Range_Anxiety_Level'),
                    ('Range_Anxiety_Level','Subsidy_Available'),('Home_Charging_Possible','City_Type')]:
            x[a+'__'+b]=x[a]+'|'+x[b]
        return x
    def fit(self,df):
        self.numeric=['Annual_Income_USD','Daily_Commute_km','Environmental_Concern_Level','Age','Charging_Stations_Near_Home','Charging_Stations_Near_Work']
        self.onehot=OneHotEncoder(handle_unknown='ignore',dtype=np.float32)
        self.onehot.fit(self.keys(df))
        self.spline=SplineTransformer(n_knots=7,degree=3,include_bias=False,extrapolation='linear')
        smooth=self.spline.fit_transform(df[self.numeric])
        self.scale=StandardScaler().fit(smooth)
        return self
    def transform(self,df):
        cat=self.onehot.transform(self.keys(df))
        smooth=self.scale.transform(self.spline.transform(df[self.numeric])).astype(np.float32)
        return hstack([cat,csr_matrix(smooth)],format='csr',dtype=np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--folds',default='0');p.add_argument('--c',type=float,default=.1);p.add_argument('--iterations',type=int,default=2000);a=p.parse_args()
    data=Path.home()/'.cache/kaggle/playground-series-s6e9';df=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');y=df[TARGET].eq('Yes').astype(int)
    out=Path(a.output);out.mkdir(exist_ok=True,parents=True);splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y))
    for f in map(int,a.folds.split(',')):
        if (out/f'fold{f}.json').exists():continue
        start=time.monotonic();tr,va=splits[f];enc=ValueEncoder().fit(df.iloc[tr]);xt=enc.transform(df.iloc[tr]);xv=enc.transform(df.iloc[va]);xx=enc.transform(test)
        print(f'fold {f} shape {xt.shape} nnz {xt.nnz} encoding {time.monotonic()-start:.1f}s',flush=True)
        model=LogisticRegression(C=a.c,max_iter=a.iterations,tol=1e-5,solver='lbfgs',random_state=42)
        with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(xt,y.iloc[tr])
        pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xx)[:,1]
        np.savez_compressed(out/f'fold{f}.npz',indices=va,oof=pv,test=pt);joblib.dump({'encoder':enc,'model':model},out/f'fold{f}.joblib')
        result={'fold':f,**score(y.iloc[va],pv),'iterations':int(model.n_iter_[0]),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in caught),'seconds':time.monotonic()-start,'C':a.c,'source_sha256':digest(__file__)}
        (out/f'fold{f}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if all((out/f'fold{f}.npz').exists() for f in range(5)):
        oof=np.full(len(df),np.nan);pred=np.zeros(len(test));folds=np.full(len(df),-1);results=[]
        for f in range(5):
            z=np.load(out/f'fold{f}.npz');oof[z['indices']]=z['oof'];pred+=z['test']/5;folds[z['indices']]=f
            results.append(json.loads((out/f'fold{f}.json').read_text()))
        pd.DataFrame({'id':df.id,'target':y,'fold':folds,'prediction':oof}).to_csv(out/'oof.csv',index=False)
        sub=pd.DataFrame({'id':test.id,TARGET:pred});validate_submission(sub,test,{'id':'id','target':TARGET});sub.to_csv(out/'submission.csv',index=False)
        summary={'args':vars(a),'oof':score(y,oof),'mean_auc':float(np.mean([r['auc'] for r in results])),'folds':results,'train_sha256':digest(data/'train.csv')}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',summary['oof'],flush=True)

if __name__=='__main__':
    from src.sparse_values import main as entry
    entry()
