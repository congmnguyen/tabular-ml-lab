import joblib
import numpy as np
import pandas as pd
from scipy.special import logit,ndtr
from src.mechanistic_encoding import fit_effect
from src.offset_boost import OffsetXGB,design_encoded,KEYS


def test_risk_adjustment_removes_explained_group_rate_difference():
    keys=pd.Series(['high']*400+['low']*400)
    y=np.r_[np.tile([1,1,1,0],100),np.tile([1,0,0,0],100)]
    margin=logit(np.r_[np.full(400,.75),np.full(400,.25)])
    effects=fit_effect(keys,y,margin)
    np.testing.assert_allclose(effects.to_numpy(),0,atol=1e-10)
    unexplained=fit_effect(keys,y,np.full(800,logit(.5)))
    assert unexplained['high']>0 and unexplained['low']<0


def test_saved_offset_model_includes_margin_at_inference(tmp_path):
    rng=np.random.default_rng(42);n=600
    x=pd.DataFrame({'Annual_Income_USD':rng.uniform(30000,160000,n),
                    'Environmental_Concern_Level':rng.integers(1,6,n),
                    KEYS[0]:rng.integers(0,2,n),KEYS[1]:rng.integers(0,2,n),KEYS[2]:np.zeros(n)})
    p=ndtr(design_encoded(x)@np.array([-5.5,1.2,.6,2,-1,-3]))
    y=rng.binomial(1,p)
    model=OffsetXGB(n_estimators=5,max_depth=2,n_jobs=1,tree_method='hist')
    model.fit(x.iloc[:500],y[:500])
    expected=model.predict_proba(x.iloc[500:])
    path=tmp_path/'offset.joblib';joblib.dump(model,path);loaded=joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(x.iloc[500:]),expected)
    from xgboost import XGBClassifier
    without_margin=XGBClassifier.predict_proba(loaded,x.iloc[500:].drop(columns=KEYS))
    assert np.max(np.abs(without_margin-expected))>.01
