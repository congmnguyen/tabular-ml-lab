"""Discover coarse shared error regions, then check a fixed correction on holdout."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.tree import DecisionTreeRegressor, export_text
from . import expert
from .low_smoothing import encoder_factory
from .modulo_features import ModuloEncoder
from .offset_boost import OffsetXGB
from .investigate import DATA, ORIG, recipe
from .pipeline import score, digest


def main():
    out=Path('artifacts/r8-shared-errors');out.mkdir(parents=True,exist_ok=True)
    report=Path('reports/round8');report.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/'train.csv');original=pd.read_csv(ORIG)
    folds=pd.read_csv('artifacts/r4-offset/oof.csv');assert df.id.equals(folds.id)
    y=df.Will_Buy_EV.eq('Yes').astype(int);train=folds.fold.le(2);heldout=~train
    if not (out/'heldout.csv').exists():
        expert.TargetEncoder=encoder_factory
        encoder=ModuloEncoder('full');xt=encoder.fit_transform(df[train],y[train],original);xh=encoder.transform(df[heldout])
        model=OffsetXGB(n_estimators=600,learning_rate=.03,max_depth=5,subsample=.9,colsample_bytree=.8,reg_lambda=10.,reg_alpha=1.,min_child_weight=5,max_bin=512,device='cuda',tree_method='hist',eval_metric='auc',n_jobs=8,random_state=42)
        model.fit(xt,y[train]);pred=model.predict_proba(xh)[:,1]
        pd.DataFrame({'id':df.id[heldout],'target':y[heldout],'fold':folds.fold[heldout],'prediction':pred}).to_csv(out/'heldout.csv',index=False)
        joblib.dump({'encoder':encoder,'model':model},out/'model.joblib')
        del xt,xh,encoder,model
    current=pd.read_csv(out/'heldout.csv');old=pd.read_csv('artifacts/r5-frozen-diagnostic/heldout.csv')
    assert np.array_equal(current[['id','target','fold']],old[['id','target','fold']])
    h=df.set_index('id').loc[current.id].reset_index()
    x=h.drop(columns=['id','Will_Buy_EV']).copy()
    x['recipe']=recipe(h);x['income_mod1000']=h.Annual_Income_USD.astype(int)%1000
    x['income_training_count']=h.Annual_Income_USD.map(df[train].Annual_Income_USD.value_counts()).fillna(0)
    source_mean=original.assign(_y=original.Will_Buy_EV.eq('Yes')).groupby('Annual_Income_USD')._y.mean()
    x['source_income_mean']=h.Annual_Income_USD.map(source_mean).fillna(source_mean.mean())
    x=pd.get_dummies(x,dtype=float).astype(float).fillna(0)
    discovery=current.fold.eq(3).to_numpy();confirmation=~discovery
    p=current.prediction.to_numpy();target=current.target.to_numpy();variance=np.clip(p*(1-p),1e-5,None)
    residual=target-p
    tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=3000,random_state=42)
    tree.fit(x[discovery],residual[discovery]/variance[discovery],sample_weight=variance[discovery])
    leaves=tree.apply(x);effects={};groups=[]
    for leaf in np.unique(leaves[discovery]):
        mask=discovery&(leaves==leaf);effect=float(residual[mask].sum()/(variance[mask].sum()+20));effects[int(leaf)]=effect
        group={'leaf':int(leaf),'discovery_rows':int(mask.sum()),'discovery_effect':effect}
        cmask=confirmation&(leaves==leaf)
        for name,pred in [('current',p),('older',old.prediction.to_numpy())]:
            v=pred*(1-pred);r=target-pred
            group[name+'_confirmation_effect']=float(r[cmask].sum()/(v[cmask].sum()+20))
        group['confirmation_rows']=int(cmask.sum())
        group['confirmation_positive_count']=int(target[cmask].sum())
        group['confirmation_observed_rate']=float(target[cmask].mean())
        group['confirmation_predicted_rate']=float(p[cmask].mean())
        groups.append(group)
    correction=np.array([effects[int(k)] for k in leaves]);adjusted=expit(logit(np.clip(p,1e-7,1-1e-7))+.5*np.clip(correction,-1,1))
    baseline=score(target[confirmation],p[confirmation]);corrected=score(target[confirmation],adjusted[confirmation])
    result={'training_folds':[0,1,2],'discovery_fold':3,'confirmation_fold':4,'fixed_model_trees':600,'tree_depth':3,'minimum_leaf_rows':3000,'leaf_ridge':20,'correction_step':.5,'baseline_confirmation':baseline,'corrected_confirmation':corrected,'auc_gain':corrected['auc']-baseline['auc'],'leaves':groups,'source_sha256':digest(__file__),'caveat':'Both frozen models exclude discovery and confirmation labels from training. These diagnostic holdouts were examined in previous rounds and are not pristine final tests. Tree structure and corrections are fitted only on discovery. One fixed diagnostic configuration; no confirmation-driven tree tuning.'}
    (report/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (report/'discovery-tree.txt').write_text(export_text(tree,feature_names=list(x),decimals=5))
    joblib.dump({'tree':tree,'columns':list(x),'effects':effects},out/'diagnostic-tree.joblib')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
