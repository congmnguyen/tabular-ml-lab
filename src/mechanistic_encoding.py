"""Cross-fit regularized exact-value log-odds effects conditional on purchase risk."""
import numpy as np
import pandas as pd
from scipy.special import ndtr,expit,logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from . import expert
from .offset_boost import OffsetEncoder,OffsetXGB
from .mechanism import design,fit_probit


def fit_effect(keys,y,margin,penalty=5.):
    codes,levels=pd.factorize(keys,sort=True)
    effects=np.zeros(len(levels))
    for _ in range(15):
        p=expit(margin+effects[codes])
        gradient=np.bincount(codes,weights=y-p,minlength=len(levels))-penalty*effects
        curvature=np.bincount(codes,weights=p*(1-p),minlength=len(levels))+penalty
        step=np.clip(gradient/curvature,-1,1);effects+=step
        if np.max(np.abs(step))<1e-6:break
    return pd.Series(effects,index=levels)

class MechanisticEncoder(OffsetEncoder):
    def fit_transform(self,df,y,original):
        base=super().fit_transform(df,y,original)
        self.effect_keys=['Annual_Income_USD','Daily_Commute_km']
        labels=np.asarray(y,dtype=float);x=design(df)
        encoded=np.zeros((len(df),len(self.effect_keys)),dtype=np.float32)
        with threadpool_limits(limits=4):
            for train,valid in StratifiedKFold(5,shuffle=True,random_state=42).split(df,labels):
                coefficients=fit_probit(x[train],labels[train])
                margin=logit(np.clip(ndtr(x[train]@coefficients),1e-7,1-1e-7))
                for j,col in enumerate(self.effect_keys):
                    keys=df[col].astype(str)
                    mapping=fit_effect(keys.iloc[train],labels[train],margin)
                    encoded[valid,j]=keys.iloc[valid].map(mapping).fillna(0)
            coefficients=fit_probit(x,labels);margin=logit(np.clip(ndtr(x@coefficients),1e-7,1-1e-7))
            self.effect_maps={c:fit_effect(df[c].astype(str),labels,margin) for c in self.effect_keys}
        for j,col in enumerate(self.effect_keys):base[col+'_risk_adjusted_effect']=encoded[:,j]
        return base
    def transform(self,df):
        base=super().transform(df)
        for col,mapping in self.effect_maps.items():base[col+'_risk_adjusted_effect']=df[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.float32)
        return base


def main():
    expert.ExpertEncoder=MechanisticEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.mechanistic_encoding import main as entry
    entry()
