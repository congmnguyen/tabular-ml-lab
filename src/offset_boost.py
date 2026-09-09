"""Boost residual log-odds around a fold-trained probit purchase mechanism."""
import numpy as np
from scipy.special import ndtr,logit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from . import expert
from .mechanism import fit_probit

KEYS=['offset_subsidy','offset_medium','offset_high']

class OffsetEncoder(expert.ExpertEncoder):
    def attach(self,df,x):
        x[KEYS[0]]=df.Subsidy_Available.eq('Yes').astype(np.float32)
        x[KEYS[1]]=df.Range_Anxiety_Level.eq('Medium').astype(np.float32)
        x[KEYS[2]]=df.Range_Anxiety_Level.eq('High').astype(np.float32)
        return x
    def fit_transform(self,df,y,original):
        return self.attach(df,super().fit_transform(df,y,original))
    def transform(self,df):
        return self.attach(df,super().transform(df))


def design_encoded(x):
    return np.column_stack([np.ones(len(x)),x.Annual_Income_USD/1e5,x.Environmental_Concern_Level,x[KEYS]]).astype(float)

class OffsetXGB(XGBClassifier):
    def margin(self,x):
        return logit(np.clip(ndtr(design_encoded(x)@self.probit_coefficients_),1e-7,1-1e-7)).astype(np.float32)
    def fit(self,x,y,*,eval_set=None,**kwargs):
        with threadpool_limits(limits=4):self.probit_coefficients_=fit_probit(design_encoded(x),np.asarray(y))
        margin=self.margin(x)
        eval_margin=[self.margin(vx) for vx,vy in eval_set] if eval_set else None
        filtered=[(vx.drop(columns=KEYS),vy) for vx,vy in eval_set] if eval_set else None
        return super().fit(x.drop(columns=KEYS),y,base_margin=margin,eval_set=filtered,base_margin_eval_set=eval_margin,**kwargs)
    def predict_proba(self,x,**kwargs):
        return super().predict_proba(x.drop(columns=KEYS),base_margin=self.margin(x),**kwargs)


def main():
    expert.ExpertEncoder=OffsetEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.offset_boost import main as entry
    entry()
