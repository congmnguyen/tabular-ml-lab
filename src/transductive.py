"""Label-free train+test frequency ablation, explicitly transductive."""
from pathlib import Path
import pandas as pd
from . import expert
from .lean import LeanEncoder
from .fine_bins import fine_model

class TransductiveEncoder(LeanEncoder):
    def fit_transform(self,df,y,original):
        base=super().fit_transform(df,y,original)
        data=Path.home()/'.cache/kaggle/playground-series-s6e9'
        # Labels are excluded at read time. TE above remains confined to outer train labels.
        unlabelled=pd.concat([pd.read_csv(data/name,usecols=lambda c:c!='Will_Buy_EV') for name in ['train.csv','test.csv']],ignore_index=True)
        global_x=self.prepare(unlabelled)
        self.freq={c:global_x[c].value_counts(normalize=True).to_dict() for c in self.columns}
        local_x=self.prepare(df)
        for c in self.columns:
            base[c+'_frequency']=local_x[c].map(self.freq[c]).fillna(0).to_numpy(dtype='float32')
        return base

def main():
    expert.ExpertEncoder=TransductiveEncoder;expert.LGBMClassifier=fine_model;expert.main()
if __name__=='__main__':
    from src.transductive import main as entry
    entry()
