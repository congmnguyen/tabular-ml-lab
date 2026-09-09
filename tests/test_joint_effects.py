"""A correlated proxy must not inherit another variable's effect after joint fitting."""
import numpy as np
import pandas as pd
from src.joint_effects import fit_joint
from src.mechanistic_encoding import fit_effect


def test_joint_fit_separates_correlated_proxy(monkeypatch):
    # Purchase rate depends only on commute, while income correlates with commute.
    blocks=[];labels=[]
    for income,commute,n,positive in [(30000,10,4500,1125),(30000,20,500,375),(50000,10,500,125),(50000,20,4500,3375)]:
        blocks.append(pd.DataFrame({'Annual_Income_USD':np.full(n,income),'Daily_Commute_km':np.full(n,commute),'Environmental_Concern_Level':np.full(n,3),'Subsidy_Available':['No']*n,'Range_Anxiety_Level':['Low']*n}))
        labels.append(np.r_[np.ones(positive),np.zeros(n-positive)])
    df=pd.concat(blocks,ignore_index=True);y=np.concatenate(labels)
    monkeypatch.setattr('src.joint_effects.fit_probit',lambda x,y:np.zeros(x.shape[1]))
    joint=fit_joint(df,y)
    independent=fit_effect(df.Annual_Income_USD.astype(str),y,np.zeros(len(y)),penalty=2.)
    assert independent.abs().max()>.5
    assert joint['Annual_Income_USD'].abs().max()<.02
    assert joint['Daily_Commute_km']['10']<-.9
    assert joint['Daily_Commute_km']['20']>.9
