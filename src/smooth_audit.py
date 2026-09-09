"""Test smooth income residual corrections on the two frozen-model holdouts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from .investigate import DATA

class IncomeBasis:
    def fit(self,df):
        self.spline=SplineTransformer(n_knots=8,degree=3,include_bias=True,extrapolation='constant').fit(df[['Annual_Income_USD']])
        return self
    def transform(self,df):
        b=self.spline.transform(df[['Annual_Income_USD']])
        return np.column_stack([b]+[b*df.Subsidy_Available.eq(v).to_numpy()[:,None] for v in ['No','Yes']]+
                               [b*df.Environmental_Concern_Level.eq(v).to_numpy()[:,None] for v in range(1,6)])


def main():
    pred=pd.read_csv('artifacts/r5-frozen-diagnostic/heldout.csv');df=pd.read_csv(DATA/'train.csv').set_index('id').loc[pred.id].reset_index()
    y=pred.target.to_numpy();p=np.clip(pred.prediction.to_numpy(),1e-7,1-1e-7);results=[]
    with threadpool_limits(limits=4):
        for fit_fold in [3,4]:
            train=pred.fold.eq(fit_fold).to_numpy();valid=~train
            basis=IncomeBasis().fit(df[train]);x=basis.transform(df[train]);xv=basis.transform(df[valid]);w=p[train]*(1-p[train])
            for penalty in [10.,100.,1000.]:
                beta=np.linalg.solve(x.T@(x*w[:,None])+penalty*np.eye(x.shape[1]),x.T@(y[train]-p[train]))
                correction=np.clip(xv@beta,-1,1);pp=expit(logit(p[valid])+correction)
                before=float(roc_auc_score(y[valid],p[valid]));after=float(roc_auc_score(y[valid],pp))
                row={'fit_fold':fit_fold,'evaluation_fold':7-fit_fold,'penalty':penalty,'baseline_auc':before,'corrected_auc':after,'auc_gain':after-before,
                     'baseline_logloss':log_loss(y[valid],p[valid]),'corrected_logloss':log_loss(y[valid],pp),'max_abs_logodds_correction':float(np.abs(correction).max())}
                results.append(row);print(row,flush=True)
    Path('reports/round6/smooth-audit.json').write_text(json.dumps({'results':results,'basis':'8-knot cubic income splines with subsidy and concern interactions',
        'warning':'Frozen base model fits neither diagnostic holdout. These repeatedly explored holdouts are development data, not a pristine final test.'},indent=2)+'\n')

if __name__=='__main__':main()
