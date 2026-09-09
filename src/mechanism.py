"""Audit the original probit mechanism and remaining cross-fold errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr
from sklearn.metrics import roc_auc_score, log_loss
from .investigate import recipe, DATA, ORIG


def design(df):
    return np.column_stack([np.ones(len(df)),df.Annual_Income_USD/1e5,
        df.Environmental_Concern_Level,df.Subsidy_Available.eq('Yes'),
        df.Range_Anxiety_Level.eq('Medium'),df.Range_Anxiety_Level.eq('High')]).astype(float)


def fit_probit(x,y):
    def objective(b):
        z=x@b; lp=log_ndtr(z); ln=log_ndtr(-z); logpdf=-z*z/2-.5*np.log(2*np.pi)
        loss=-np.mean(y*lp+(1-y)*ln)
        grad=x.T@np.where(y,-np.exp(logpdf-lp),np.exp(logpdf-ln))/len(y)
        return loss+1e-6*np.sum(b[1:]**2),grad+np.r_[0,2e-6*b[1:]]
    r=minimize(objective,[-5.5,1.2,.6,2.,-1.,-3.],jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12})
    if not r.success:raise RuntimeError(r.message)
    return r.x


def main():
    train=pd.read_csv(DATA/'train.csv');orig=pd.read_csv(ORIG).dropna(subset=['Annual_Income_USD','Environmental_Concern_Level'])
    oof=pd.read_csv('artifacts/r2-full-xgb/oof.csv');assert train.id.equals(oof.id)
    y=train.Will_Buy_EV.eq('Yes').to_numpy();fold=oof.fold.to_numpy();x=design(train)
    pred=np.zeros(len(y));coefs=[]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        for f in range(5):
            b=fit_probit(x[fold!=f],y[fold!=f]);pred[fold==f]=ndtr(x[fold==f]@b);coefs.append(b.tolist())
        original_coefficients=fit_probit(design(orig),orig.Will_Buy_EV.eq('Yes').to_numpy()).tolist()
    tables={}
    for col in ['Annual_Income_USD','Environmental_Concern_Level','Range_Anxiety_Level','Subsidy_Available']:
        key=pd.qcut(train[col],10,duplicates='drop').astype(str) if col=='Annual_Income_USD' else train[col]
        rows=pd.DataFrame({'key':key,'y':y,'residual':y-oof.prediction,'fold':fold})
        agg=rows.groupby(['key','fold'],observed=True).agg(n=('y','size'),residual=('residual','mean')).reset_index()
        tables[col]=agg.to_dict('records')
    out=Path('reports/round4');out.mkdir(exist_ok=True,parents=True)
    result={'columns':['intercept','income/100000','concern','subsidy_yes','anxiety_medium','anxiety_high'],
        'original_recipe_coefficients':[-5.5,1.2,.6,2,-1,-3],'original_fitted_coefficients':original_coefficients,
        'synthetic_fold_coefficients':coefs,'fixed_recipe_auc':roc_auc_score(y,recipe(train)),
        'fitted_probit_oof_auc':roc_auc_score(y,pred),'fitted_probit_oof_logloss':log_loss(y,np.clip(pred,1e-8,1-1e-8)),
        'current_xgb_oof_auc':roc_auc_score(y,oof.prediction),'residual_by_group_and_fold':tables}
    (out/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='residual_by_group_and_fold'},indent=2))

if __name__=='__main__':main()
