"""Condition repeated-value effects on the variables in the source buying mechanism."""
import numpy as np
from . import expert
from .expert import ExpertEncoder

class ConditionalEncoder(ExpertEncoder):
    def prepare(self,df):
        x=super().prepare(df)
        pairs=[('Annual_Income_USD','Subsidy_Available'),('Annual_Income_USD','Range_Anxiety_Level'),
               ('Annual_Income_USD','Home_Charging_Possible'),('income_bin100','Environmental_Concern_Level'),
               ('income_bin1000','Environmental_Concern_Level'),('income_bin1000','Subsidy_Available'),
               ('Daily_Commute_km','Home_Charging_Possible'),('commute_bin1','Range_Anxiety_Level'),
               ('Environmental_Concern_Level','Subsidy_Available'),('Environmental_Concern_Level','Range_Anxiety_Level')]
        for a,b in pairs:x[a+'__'+b]=x[a].fillna('__NA__').astype(str)+'|'+x[b].fillna('__NA__').astype(str)
        return x

def main():
    expert.ExpertEncoder=ConditionalEncoder;expert.main()
if __name__=='__main__':
    from src.conditional import main as entry
    entry()
