"""Ablate high-cardinality recipe keys; retain raw, digit and multi-scale keys."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from . import expert
from .expert import ExpertEncoder
from .fine_bins import fine_model

class LeanEncoder(ExpertEncoder):
    def prepare(self,df):
        x=super().prepare(df)
        drop=[c for c in x if c in ['recipe','worry','subsidy_income','subsidy_concern'] or c.startswith('Number_of_Cars_Owned')]
        return x.drop(columns=drop)
    def fit_transform(self,df,y,original):
        base=super().fit_transform(df,y,original)
        self.auto=TargetEncoder(target_type='binary',smooth='auto',cv=5,shuffle=True,random_state=self.seed)
        added=self.auto.fit_transform(self.key_frame(self.prepare(df)),y)
        return self.join(base,added)
    def join(self,base,added):
        base=base.drop(columns=['Number_of_Cars_Owned_original_mean'],errors='ignore')
        return pd.concat([base,pd.DataFrame(added,index=base.index,columns=[c+'_te_auto' for c in self.keys],dtype=np.float32)],axis=1)
    def transform(self,df):
        return self.join(super().transform(df),self.auto.transform(self.key_frame(self.prepare(df))))

def main():
    expert.ExpertEncoder=LeanEncoder;expert.LGBMClassifier=fine_model;expert.main()
if __name__=='__main__':
    from src.lean import main as entry
    entry()
