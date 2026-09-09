"""Unlabeled source-row context for uniquely matched income values."""
import numpy as np
import pandas as pd
from . import expert
from .low_smoothing import encoder_factory
from .modulo_features import ModuloEncoder
from .offset_boost import OffsetXGB
from .investigate import recipe

COLUMNS=['Age','Gender','City_Type','Daily_Commute_km','Number_of_Cars_Owned','Current_Car_Type','Charging_Stations_Near_Home','Charging_Stations_Near_Work','Home_Charging_Possible','Environmental_Concern_Level','Subsidy_Available','Range_Anxiety_Level']


class SourceContextEncoder(ModuloEncoder):
    def fit_transform(self,df,y,original):
        eligible=original.Annual_Income_USD.notna() & ~original.Annual_Income_USD.duplicated(keep=False)
        source=original.loc[eligible,['Annual_Income_USD',*COLUMNS]].copy()
        source['source_recipe']=recipe(source)
        self.source_maps={c:source.set_index('Annual_Income_USD')[c].to_dict() for c in [*COLUMNS,'source_recipe']}
        return self.attach_source(df,super().fit_transform(df,y,original))

    def transform(self,df):
        return self.attach_source(df,super().transform(df))

    def attach_source(self,df,base):
        mapped={c:df.Annual_Income_USD.map(m) for c,m in self.source_maps.items()}
        known=df.Annual_Income_USD.isin(self.source_maps['Age'])
        matches=np.zeros(len(df));observed=np.zeros(len(df))
        for c in COLUMNS:
            valid=known & mapped[c].notna() & df[c].notna()
            equal=valid & df[c].eq(mapped[c]);matches+=equal.to_numpy();observed+=valid.to_numpy()
            if c in ['Current_Car_Type','City_Type','Home_Charging_Possible','Range_Anxiety_Level','Subsidy_Available']:
                base['source_match_'+c]=equal.to_numpy(dtype=np.float32)
        base['source_unique_income']=known.to_numpy(dtype=np.float32)
        base['source_match_fraction']=(matches/np.maximum(observed,1)).astype(np.float32)
        for c in ['Daily_Commute_km','Environmental_Concern_Level','Charging_Stations_Near_Home','Charging_Stations_Near_Work']:
            base['source_delta_'+c]=(df[c]-mapped[c]).fillna(0).to_numpy(dtype=np.float32)
        base['source_recipe']=mapped['source_recipe'].fillna(0).to_numpy(dtype=np.float32)
        base['source_delta_recipe']=(recipe(df)-mapped['source_recipe']).fillna(0).to_numpy(dtype=np.float32)
        return base


def main():
    expert.TargetEncoder=encoder_factory
    expert.ExpertEncoder=SourceContextEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.source_context import main as entry
    entry()
