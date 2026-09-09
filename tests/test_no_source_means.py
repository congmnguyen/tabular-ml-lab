"""Removing source means must make features invariant to public source labels."""
import numpy as np
import pandas as pd
from src.no_source_means import NoSourceMeansEncoder
from src.low_smoothing import encoder_factory


def test_removed_source_labels_cannot_affect_features(monkeypatch):
    n=30
    df=pd.DataFrame({'id':range(n),'Age':np.arange(n)+25,'Annual_Income_USD':np.arange(n)*1000.+30000,
        'Daily_Commute_km':np.arange(n)*.1+5,'Number_of_Cars_Owned':[1]*n,
        'Charging_Stations_Near_Home':[2]*n,'Charging_Stations_Near_Work':[3]*n,
        'Environmental_Concern_Level':[3.]*n,'Gender':['Male','Female']*(n//2),
        'City_Type':['Urban']*n,'Current_Car_Type':['Sedan']*n,'Home_Charging_Possible':['Yes']*n,
        'Subsidy_Available':['No']*n,'Range_Anxiety_Level':['Low']*n})
    y=np.tile([0,1],n//2);source=df.assign(Will_Buy_EV=np.where(y,'Yes','No'))
    flipped=source.assign(Will_Buy_EV=np.where(y,'No','Yes'))
    monkeypatch.setattr('src.expert.TargetEncoder',encoder_factory)
    a=NoSourceMeansEncoder();b=NoSourceMeansEncoder()
    first=a.fit_transform(df,y,source);second=b.fit_transform(df,y,flipped)
    pd.testing.assert_frame_equal(first,second)
    pd.testing.assert_frame_equal(a.transform(df),b.transform(df))
    assert not any(c.endswith('_original_mean') for c in first)
    assert 'offset_subsidy' in first
