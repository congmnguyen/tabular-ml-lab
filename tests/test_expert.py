import numpy as np
import pandas as pd
from src.expert import ExpertEncoder

def test_target_encoding_is_cross_fitted_and_unseen_is_finite():
    n=50
    df=pd.DataFrame({'id':range(n),'Age':range(25,25+n),'Annual_Income_USD':np.arange(n)*1000.+30000,
        'Daily_Commute_km':np.arange(n)*.1+5,'Number_of_Cars_Owned':[1]*n,
        'Charging_Stations_Near_Home':[2]*n,'Charging_Stations_Near_Work':[3]*n,
        'Environmental_Concern_Level':[3.]*n,'Gender':['Male','Female']*(n//2),
        'City_Type':['Urban']*n,'Current_Car_Type':['Sedan']*n,'Home_Charging_Possible':['Yes']*n,
        'Subsidy_Available':['No']*n,'Range_Anxiety_Level':['Low']*n})
    y=np.array([0,1]*(n//2)); orig=df.assign(Will_Buy_EV=np.where(y,'Yes','No'))
    encoder=ExpertEncoder('full')
    x=encoder.fit_transform(df,y,orig)
    # Each age is unique: its training TE must be the inner-train prior, not its own label.
    np.testing.assert_allclose(x.Age_te0,.5)
    unseen=df.iloc[:1].copy();unseen['Age']=999;unseen['Gender']='Unseen'
    p=encoder.transform(unseen)
    assert list(p)==list(x)
    assert np.isfinite(p.to_numpy()).all()
    assert p.Gender.iloc[0]==-1
    np.testing.assert_allclose(p.Age_te0,.5)

def test_inference_preserves_training_fold_accumulation(monkeypatch):
    from src import predict_expert
    arrays=[np.array([.1234567+i*.0123456,.9876543-i*.013579],dtype=np.float32) for i in range(5)]
    class Encoder:
        def transform(self,frame):return frame
    class Model:
        def __init__(self,probs):self.probs=probs
        def predict_proba(self,frame):return np.column_stack([1-self.probs,self.probs])
    def load(path):
        i=int(path.stem.removeprefix('fold'))
        return {'encoder':Encoder(),'model':Model(arrays[i])}
    monkeypatch.setattr(predict_expert.joblib,'load',load)
    expected=np.zeros(2,dtype=np.float64)
    for p in arrays:expected+=p/5
    np.testing.assert_array_equal(predict_expert.predict_run('unused',pd.DataFrame({'id':[1,2]})),expected)
