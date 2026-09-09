"""Screen higher histogram resolution for repeated-income effects (see research.md)."""
from . import expert
from lightgbm import LGBMClassifier


def fine_model(**params):
    params.update(max_bin=1024, colsample_bytree=.3, min_child_samples=10, reg_lambda=2.)
    return LGBMClassifier(**params)


def main():
    expert.LGBMClassifier=fine_model
    expert.main()

if __name__=='__main__':
    from src.fine_bins import main as entry
    entry()
