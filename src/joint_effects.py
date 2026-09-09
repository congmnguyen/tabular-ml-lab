"""Jointly estimate income and commute effects around fold-fitted purchase risk."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from . import expert
from .low_smoothing import encoder_factory
from .modulo_features import ModuloEncoder
from .offset_boost import OffsetXGB
from .mechanism import design, fit_probit

KEYS=['Annual_Income_USD','Daily_Commute_km']


def fit_joint(df,y,penalty=2.,iterations=30):
    with threadpool_limits(limits=4):
        coef=fit_probit(design(df),y)
    margin=logit(np.clip(ndtr(design(df)@coef),1e-7,1-1e-7))
    coding=[pd.factorize(df[c].astype(str),sort=True) for c in KEYS]
    effects=[np.zeros(len(levels)) for codes,levels in coding]
    for _ in range(iterations):
        max_step=0.
        for j,(codes,levels) in enumerate(coding):
            p=expit(margin)
            gradient=np.bincount(codes,weights=y-p,minlength=len(levels))-penalty*effects[j]
            curvature=np.bincount(codes,weights=p*(1-p),minlength=len(levels))+penalty
            step=np.clip(gradient/curvature,-.5,.5)
            effects[j]+=step;margin+=step[codes];max_step=max(max_step,float(np.max(np.abs(step))))
        if max_step<1e-5:break
    return {c:pd.Series(e,index=levels) for c,e,(codes,levels) in zip(KEYS,effects,coding)}


class JointEncoder(ModuloEncoder):
    def fit_transform(self,df,y,original):
        base=super().fit_transform(df,y,original)
        labels=np.asarray(y,dtype=float);encoded=np.zeros((len(df),len(KEYS)),dtype=np.float32)
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(df,labels):
            maps=fit_joint(df.iloc[tr],labels[tr])
            for j,c in enumerate(KEYS):encoded[va,j]=df.iloc[va][c].astype(str).map(maps[c]).fillna(0)
        self.joint_maps=fit_joint(df,labels)
        for j,c in enumerate(KEYS):base[c+'_joint_effect']=encoded[:,j]
        return base

    def transform(self,df):
        base=super().transform(df)
        for c,mapping in self.joint_maps.items():base[c+'_joint_effect']=df[c].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.float32)
        return base


def main():
    expert.TargetEncoder=encoder_factory
    expert.ExpertEncoder=JointEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.joint_effects import main as entry
    entry()
