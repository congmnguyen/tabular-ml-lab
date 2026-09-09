"""Explore whether income-specific OOF errors repeat across disjoint row groups."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .investigate import DATA


def stats(df,keys):
    return df.groupby(keys,observed=True).agg(n=('residual','size'),r=('residual','sum'),v=('variance','sum'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['oof','frozen'],default='frozen');args=parser.parse_args()
    source='artifacts/r4-offset/oof.csv' if args.source=='oof' else 'artifacts/r5-frozen-diagnostic/heldout.csv'
    df=pd.read_csv(DATA/'train.csv');oof=pd.read_csv(source)
    df=df.set_index('id').loc[oof.id].reset_index();assert df.id.equals(oof.id)
    discovery_folds=[0,1] if args.source=='oof' else [3]
    confirmation_folds=[2,3,4] if args.source=='oof' else [4]
    df['residual']=oof.target-oof.prediction;df['variance']=oof.prediction*(1-oof.prediction);df['fold']=oof.fold
    discovery=df[df.fold.isin(discovery_folds)];confirmation=df[df.fold.isin(confirmation_folds)]
    results={}
    for group in ['Subsidy_Available','Environmental_Concern_Level']:
        pieces=[]
        for part in [discovery,confirmation]:
            part=part.copy()
            group_stats=stats(part,[group]);group_beta=group_stats.r/group_stats.v.clip(lower=1e-8)
            part['residual']=part.residual-part.variance*part[group].map(group_beta)
            parent=stats(part,['Annual_Income_USD']);child=stats(part,['Annual_Income_USD',group])
            parent_beta=parent.r/parent.v.clip(lower=1e-8)
            child['parent_beta']=child.index.get_level_values(0).map(parent_beta)
            child['interaction_score']=child.r-child.v*child.parent_beta
            child['effect']=child.interaction_score/(child.v+20.)
            child['z']=child.interaction_score/np.sqrt(child.v.clip(lower=1e-8))
            pieces.append(child)
        joined=pieces[0].join(pieces[1],lsuffix='_discovery',rsuffix='_confirmation')
        eligible=joined[(joined.n_discovery>=30)&(joined.n_confirmation>=30)&(joined.v_discovery>=3)&(joined.v_confirmation>=3)].copy()
        top=eligible.reindex(eligible.z_discovery.abs().sort_values(ascending=False).index).head(30)
        summary={'eligible_groups':len(eligible),'selected_groups':len(top),'discovery_rows':len(discovery),'confirmation_rows':len(confirmation),'covered_rows_discovery':int(eligible.n_discovery.sum()),
            'spearman_effect':float(spearmanr(eligible.effect_discovery,eligible.effect_confirmation).statistic),
            'sign_agreement':float((np.sign(eligible.effect_discovery)==np.sign(eligible.effect_confirmation)).mean()),
            'top30_sign_agreement':float((np.sign(top.effect_discovery)==np.sign(top.effect_confirmation)).mean()),
            'top_discovery_groups':top.reset_index().to_dict('records')}
        results[group]=summary;print(group,{k:v for k,v in summary.items() if k!='top_discovery_groups'})
    report={'source':args.source,'discovery_folds':discovery_folds,'confirmation_folds':confirmation_folds,'criteria':'At least 30 rows and variance sum >=3 on each side; select top groups on discovery only.',
        'warning':'Exploratory diagnostic; remove global subgroup and income effects before comparing interactions. OOF models overlap in training; frozen-model holdouts do not enter its fit. These residuals must not be used for global stacking.', 'results':results}
    Path(f'reports/round5/{args.source}-audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
