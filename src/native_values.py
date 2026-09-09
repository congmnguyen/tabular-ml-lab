"""Test XGBoost's categorical partitions for repeated exact numeric values."""
import pandas as pd
from xgboost import XGBClassifier
from . import expert

class NativeValueEncoder(expert.ExpertEncoder):
    def fit_transform(self,df,y,original):
        self.value_categories={c:sorted(df[c].dropna().astype(str).unique()) for c in ['Annual_Income_USD','Daily_Commute_km']}
        return self.attach(df,super().fit_transform(df,y,original))
    def attach(self,df,x):
        for c,categories in self.value_categories.items():
            x[c+'_native_value']=pd.Categorical(df[c].astype(str),categories=categories)
        return x
    def transform(self,df):
        return self.attach(df,super().transform(df))

def factory(**params):
    params.update(enable_categorical=True,max_cat_to_onehot=4,max_cat_threshold=64,device='cpu')
    return XGBClassifier(**params)

def main():
    expert.ExpertEncoder=NativeValueEncoder
    expert.XGBClassifier=factory
    expert.main()

if __name__=='__main__':
    from src.native_values import main as entry
    entry()
