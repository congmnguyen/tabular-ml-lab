"""Test joint remainders and multi-resolution income views with the offset model.
Research attribution: reports/round6/report.md.
"""
import numpy as np
from . import expert
from .offset_boost import OffsetEncoder,OffsetXGB

class ModuloEncoder(OffsetEncoder):
    def prepare(self,df):
        x=super().prepare(df)
        income=df.Annual_Income_USD.fillna(0).to_numpy().astype(np.int64)
        commute=np.rint(df.Daily_Commute_km.fillna(0).to_numpy()*10).astype(np.int64)
        x['income_mod100']=income%100;x['income_mod1000']=income%1000
        x['commute_mod100']=commute%100
        for step in [250,500,2500]:x[f'income_resolution_{step}']=income//step
        return x


def main():
    expert.ExpertEncoder=ModuloEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.modulo_features import main as entry
    entry()
