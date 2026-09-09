"""Train with a low-weight public source dataset in addition to synthetic rows."""
import numpy as np
import pandas as pd
from . import expert
from lightgbm import LGBMClassifier

_original_x = None
_original_y = None

class AugmentEncoder(expert.ExpertEncoder):
    def fit_transform(self,df,y,original):
        global _original_x, _original_y
        x=super().fit_transform(df,y,original)
        _original_x=self.transform(original)
        _original_y=original[expert.TARGET].eq('Yes').to_numpy()
        return x

class AugmentedLGBM(LGBMClassifier):
    def fit(self,x,y,**kwargs):
        augmented=pd.concat([x,_original_x],ignore_index=True)
        target=np.concatenate([y,_original_y])
        weights=np.concatenate([np.ones(len(y)),np.full(len(_original_y),.25)])
        return super().fit(augmented,target,sample_weight=weights,**kwargs)

def main():
    expert.ExpertEncoder=AugmentEncoder
    expert.LGBMClassifier=AugmentedLGBM
    expert.main()

if __name__=='__main__':
    from src.original_augmentation import main as entry
    entry()
