import joblib
import numpy as np
import pandas as pd
import pytest
from src.pipeline import build_model, predict_bundle, validate_submission, features

CONFIG = {'id':'id','target':'y','positive_label':'Yes','seed':42,'folds':2,'threads':2}

def test_fold_preprocessing_and_artifact(tmp_path):
    train = pd.DataFrame({'amount':[1.,2.,3.,4.,5.,6.], 'category':['a','b','a','b','a','b']})
    model, cats = build_model(train, 'logistic', CONFIG)
    model.fit(train, [0,1,0,1,0,1])
    valid = pd.DataFrame({'amount':[100000., np.nan], 'category':['unseen',None]})
    bundle = dict(model=model, kind='logistic', categorical=cats, columns=list(train), config=CONFIG, engineered=False)
    assert model['preprocess'].named_transformers_['numeric']['impute'].statistics_[0] == 3.5
    assert 'unseen' not in model['preprocess'].named_transformers_['categorical']['encode'].categories_[0]
    pred = predict_bundle(bundle, valid)
    assert np.isfinite(pred).all()
    joblib.dump(bundle, tmp_path/'model.joblib')
    np.testing.assert_array_equal(pred, predict_bundle(joblib.load(tmp_path/'model.joblib'), valid))
    with pytest.raises(ValueError):
        predict_bundle(bundle, valid[['category','amount']])

def test_submission_rejects_bad_order_and_probabilities():
    test = pd.DataFrame({'id':[10,11]})
    validate_submission(pd.DataFrame({'id':[10,11],'y':[.1,.9]}), test, CONFIG)
    with pytest.raises(AssertionError):
        validate_submission(pd.DataFrame({'id':[11,10],'y':[.1,.9]}), test, CONFIG)
    with pytest.raises(AssertionError):
        validate_submission(pd.DataFrame({'id':[10,11],'y':[np.nan,1.5]}), test, CONFIG)

def test_features_exclude_identifiers_and_target():
    x = features(pd.DataFrame({'id':[1],'y':['Yes'],'amount':[3.]}),CONFIG)
    assert list(x) == ['amount']
