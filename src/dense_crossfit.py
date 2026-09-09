"""Reduce the train/inference precision gap of target means with 20 inner folds."""
from sklearn.preprocessing import TargetEncoder
from . import expert
from .offset_boost import OffsetEncoder,OffsetXGB


def encoder_factory(**params):
    params['cv']=20
    return TargetEncoder(**params)


def main():
    expert.TargetEncoder=encoder_factory
    expert.ExpertEncoder=OffsetEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.dense_crossfit import main as entry
    entry()
