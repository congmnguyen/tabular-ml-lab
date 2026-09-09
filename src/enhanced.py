"""Test synthetic replication ratios and cross-fitted residual encoding."""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.preprocessing import TargetEncoder
from . import expert
from .expert import ExpertEncoder
from .investigate import recipe


class ResidualEncoder(ExpertEncoder):
    def fit_transform(self,df,y,original):
        base=super().fit_transform(df,y,original)
        x=self.prepare(df)
        self.auto=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=self.seed)
        auto=self.auto.fit_transform(self.key_frame(x),y)
        self.residual_keys=['Annual_Income_USD','Daily_Commute_km','income_bin100','income_bin1000','commute_bin1']
        self.residual=TargetEncoder(target_type='continuous',smooth=20.,cv=5,shuffle=True,random_state=self.seed)
        resid=np.asarray(y)-ndtr(recipe(df).to_numpy()-5.5)
        residual=self.residual.fit_transform(x[self.residual_keys].astype(str),resid)
        self.source_frequency={c:original[c].value_counts(normalize=True).to_dict() for c in ['Annual_Income_USD','Daily_Commute_km']}
        return self.extra(base,df,auto,residual)

    def extra(self,base,df,auto,residual):
        more={f'{c}_te_auto':auto[:,i] for i,c in enumerate(self.keys)}
        more.update({f'{c}_residual':residual[:,i] for i,c in enumerate(self.residual_keys)})
        for c,mapping in self.source_frequency.items():
            original_frequency=df[c].map(mapping).fillna(0).to_numpy()
            train_frequency=df[c].map(self.freq[c]).fillna(0).to_numpy()
            more[c+'_original_frequency']=original_frequency
            more[c+'_new_value']=(original_frequency==0).astype(float)
            more[c+'_replication_logratio']=np.log1p(train_frequency/(original_frequency+1e-5))
        return pd.concat([base,pd.DataFrame(more,index=df.index,dtype=np.float32)],axis=1)

    def transform(self,df):
        base=super().transform(df)
        x=self.prepare(df)
        auto=self.auto.transform(self.key_frame(x))
        residual=self.residual.transform(x[self.residual_keys].astype(str))
        return self.extra(base,df,auto,residual)


def main():
    expert.ExpertEncoder=ResidualEncoder
    expert.main()

if __name__=='__main__':
    from src.enhanced import main as entry
    entry()
