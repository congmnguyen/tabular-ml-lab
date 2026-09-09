"""Fold-local raw features with an optional repeated-value categorical view."""
import pandas as pd
import numpy as np
from .investigate import recipe


class RealMLPFeatures:
    def __init__(self,identities=False):
        self.identities=identities

    def prepare(self,frame):
        x=frame.drop(columns=['id','Buyer_ID','Will_Buy_EV'],errors='ignore').copy()
        x['recipe']=recipe(frame)
        x['worry']=frame.Daily_Commute_km-5*(frame.Charging_Stations_Near_Home+frame.Charging_Stations_Near_Work)-150*frame.Home_Charging_Possible.eq('Yes')
        x['income_mod1000']=(frame.Annual_Income_USD.astype(np.int64)%1000).astype(float)
        if self.identities:
            x['income_identity']=frame.Annual_Income_USD.astype(str)
            x['commute_identity']=frame.Daily_Commute_km.astype(str)
        return x

    def fit_transform(self,frame):
        x=self.prepare(frame)
        self.cat_columns=list(x.select_dtypes(include=['object','string','category']).columns)
        self.vocabularies={c:sorted(x[c].dropna().astype(str).unique().tolist()) for c in self.cat_columns}
        return self.transform(frame)

    def transform(self,frame):
        x=self.prepare(frame)
        for c,values in self.vocabularies.items():
            column=x[c].astype(str)
            x[c]=pd.Categorical(column.where(column.isin(values)),categories=values)
        return x
