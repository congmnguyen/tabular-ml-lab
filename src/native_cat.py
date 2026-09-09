"""Native categorical boosting ablation to diversify explicit target-encoded trees."""
import argparse,json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from .expert import raw_features,TARGET
from .pipeline import score,digest,validate_submission


class NativeEncoder:
    def fit(self,df,orig):
        self.num=df.drop(columns=['id',TARGET],errors='ignore').select_dtypes(include='number').columns.tolist()
        self.means={c:orig.assign(_y=orig[TARGET].eq('Yes').astype(int)).groupby(c)._y.mean().to_dict() for c in self.num}
        self.prior=float(orig[TARGET].eq('Yes').mean())
        self.freq={c:df[c].value_counts(normalize=True).to_dict() for c in self.num}
        return self
    def transform(self,df):
        x=raw_features(df,'full')
        for c in self.num:
            x[c+'_key']=df[c].fillna(-999).astype(str)
            x[c+'_orig']=df[c].map(self.means[c]).fillna(self.prior)
            x[c+'_freq']=df[c].map(self.freq[c]).fillna(0)
        for c in x:
            if not pd.api.types.is_numeric_dtype(x[c]):x[c]=x[c].fillna('__NA__').astype(str)
        return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--folds',default='0');p.add_argument('--rounds',type=int,default=3000);a=p.parse_args()
    data=Path.home()/'.cache/kaggle/playground-series-s6e9';original=Path.home()/'.cache/kaggle/ev-original/EV_Adoption_and_Range_Anxiety_Dataset.csv'
    df=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');orig=pd.read_csv(original);y=df[TARGET].eq('Yes').astype(int)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(df,y));out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    for f in map(int,a.folds.split(',')):
        if (out/f'fold{f}.json').exists():continue
        start=time.monotonic();tr,va=splits[f];enc=NativeEncoder().fit(df.iloc[tr],orig)
        xt,xv,xx=[enc.transform(part) for part in [df.iloc[tr],df.iloc[va],test]]
        cats=[c for c in xt if not pd.api.types.is_numeric_dtype(xt[c])]
        params=dict(iterations=a.rounds,depth=6,learning_rate=.04,l2_leaf_reg=5,loss_function='Logloss',eval_metric='AUC',task_type='GPU',devices='0',random_seed=42,thread_count=8,allow_writing_files=False,early_stopping_rounds=250,verbose=500)
        model=CatBoostClassifier(**params);model.fit(xt,y.iloc[tr],cat_features=cats,eval_set=(xv,y.iloc[va]))
        pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xx)[:,1]
        np.savez_compressed(out/f'fold{f}.npz',indices=va,oof=pv,test=pt)
        joblib.dump({'encoder':enc,'model':model},out/f'fold{f}.joblib')
        result={'fold':f,**score(y.iloc[va],pv),'best_iteration':model.best_iteration_+1,'seconds':time.monotonic()-start,'params':params,'source_sha256':digest(__file__)}
        (out/f'fold{f}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if all((out/f'fold{f}.npz').exists() for f in range(5)):
        oof=np.full(len(df),np.nan);pred=np.zeros(len(test));fold=np.full(len(df),-1);results=[]
        for f in range(5):
            z=np.load(out/f'fold{f}.npz');oof[z['indices']]=z['oof'];pred+=z['test']/5;fold[z['indices']]=f
            results.append(json.loads((out/f'fold{f}.json').read_text()))
        pd.DataFrame({'id':df.id,'target':y,'fold':fold,'prediction':oof}).to_csv(out/'oof.csv',index=False)
        sub=pd.DataFrame({'id':test.id,TARGET:pred});validate_submission(sub,test,{'id':'id','target':TARGET});sub.to_csv(out/'submission.csv',index=False)
        summary={'args':vars(a),'oof':score(y,oof),'mean_auc':float(np.mean([r['auc'] for r in results])),'folds':results,'gpu_nondeterministic':True,'early_stopping_on_scored_fold':True,'train_sha256':digest(data/'train.csv'),'original_sha256':digest(original)}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',summary['oof'],flush=True)

if __name__=='__main__':
    from src.native_cat import main as entry
    entry()
