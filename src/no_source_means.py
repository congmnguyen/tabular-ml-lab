"""Selected ablation: remove public-source target means from the incumbent."""
from . import expert
from .low_smoothing import encoder_factory
from .modulo_features import ModuloEncoder
from .offset_boost import OffsetXGB
from .ablation_audit import dropped_columns


class NoSourceMeansEncoder(ModuloEncoder):
    def fit_transform(self,df,y,original):
        base=super().fit_transform(df,y,original)
        self.removed_columns=dropped_columns(list(base),'source_means')
        return base.drop(columns=self.removed_columns)

    def transform(self,df):
        return super().transform(df).drop(columns=self.removed_columns)


def main():
    expert.TargetEncoder=encoder_factory
    expert.ExpertEncoder=NoSourceMeansEncoder
    expert.XGBClassifier=OffsetXGB
    expert.main()

if __name__=='__main__':
    from src.no_source_means import main as entry
    entry()
