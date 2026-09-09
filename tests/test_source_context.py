"""Source context must not choose arbitrary records for ambiguous or unseen incomes."""
import numpy as np
import pandas as pd
from src.modulo_features import ModuloEncoder
from src.source_context import SourceContextEncoder


def test_source_context_respects_unique_matches(monkeypatch):
    original=pd.DataFrame({'Annual_Income_USD':[30000.,30000.,40000.],'Age':[30,60,40],
        'Daily_Commute_km':[10.,20.,30.],'Number_of_Cars_Owned':[1,2,1],
        'Charging_Stations_Near_Home':[2,3,4],'Charging_Stations_Near_Work':[3,4,5],
        'Environmental_Concern_Level':[3.,4.,2.],'Gender':['Male','Female','Male'],
        'City_Type':['Urban']*3,'Current_Car_Type':['Sedan','SUV','Sedan'],
        'Home_Charging_Possible':['Yes']*3,'Subsidy_Available':['No']*3,'Range_Anxiety_Level':['Low']*3})
    frame=original.iloc[[0,2,2]].reset_index(drop=True);frame.loc[2,'Annual_Income_USD']=99999.
    monkeypatch.setattr(ModuloEncoder,'fit_transform',lambda self,df,y,orig:pd.DataFrame(index=df.index))
    monkeypatch.setattr(ModuloEncoder,'transform',lambda self,df:pd.DataFrame(index=df.index))
    encoder=SourceContextEncoder();result=encoder.fit_transform(frame,np.array([0,1,0]),original)
    assert result.source_unique_income.tolist()==[0.,1.,0.]
    assert result.source_match_fraction.tolist()==[0.,1.,0.]
    assert (result.iloc[[0,2]].to_numpy()==0).all()
    pd.testing.assert_frame_equal(result,encoder.transform(frame))
