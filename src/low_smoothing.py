"""Test weaker shrinkage for repeated-value target statistics with modulo features."""
from sklearn.preprocessing import TargetEncoder
from . import expert
from .modulo_features import ModuloEncoder
from .offset_boost import OffsetXGB


def encoder_factory(**params):
    params['smooth']={10.:2.,100.:20.}.get(params.get('smooth'),params.get('smooth'))
    return TargetEncoder(**params)


def main():
    expert.TargetEncoder=encoder_factory
    expert.ExpertEncoder=ModuloEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.low_smoothing import main as entry
    entry()
