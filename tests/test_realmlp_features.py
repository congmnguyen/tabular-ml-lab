import pandas as pd
from src.realmlp_features import RealMLPFeatures


def test_identity_vocabularies_are_train_local_and_preserve_numeric_view():
    frame=pd.DataFrame({'Annual_Income_USD':[30000.,40000.],'Daily_Commute_km':[10.,20.],
        'Environmental_Concern_Level':[3,4],'Subsidy_Available':['Yes','No'],
        'Range_Anxiety_Level':['Low','Medium'],'Charging_Stations_Near_Home':[1,2],
        'Charging_Stations_Near_Work':[3,4],'Home_Charging_Possible':['Yes','No']})
    plain=RealMLPFeatures(False).fit_transform(frame)
    encoder=RealMLPFeatures(True);enhanced=encoder.fit_transform(frame)
    pd.testing.assert_frame_equal(plain,enhanced.drop(columns=['income_identity','commute_identity']))
    unseen=frame.iloc[:1].copy();unseen.Annual_Income_USD=99999.;unseen.Daily_Commute_km=99.
    transformed=encoder.transform(unseen)
    assert transformed.income_identity.isna().all()
    assert transformed.commute_identity.isna().all()
    assert transformed.Annual_Income_USD.iloc[0]==99999.
    assert '99999.0' not in encoder.vocabularies['income_identity']
