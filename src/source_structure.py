"""Label-free test of source-row covariate retention beyond shared income values."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .investigate import DATA, ORIG
from .pipeline import digest

EXCLUDE={'id','Buyer_ID','Will_Buy_EV','Annual_Income_USD'}


def main():
    original=pd.read_csv(ORIG)
    source=original[original.Annual_Income_USD.notna() & ~original.Annual_Income_USD.duplicated(keep=False)].reset_index(drop=True)
    columns=[c for c in source if c not in EXCLUDE]
    mapping=pd.Series(np.arange(len(source)),index=source.Annual_Income_USD)
    results={}
    for split in ['train','test']:
        df=pd.read_csv(DATA/f'{split}.csv',usecols=lambda c:c!='Will_Buy_EV')
        index=df.Annual_Income_USD.map(mapping);eligible=index.notna();rows=df.loc[eligible,columns].copy();ix=index[eligible].to_numpy(dtype=int)
        left=[];right=[]
        for c in columns:
            if pd.api.types.is_numeric_dtype(source[c]):
                left.append(rows[c].to_numpy(dtype=float));right.append(source[c].to_numpy(dtype=float))
            else:
                values=pd.concat([source[c],rows[c]],ignore_index=True);codes,_=pd.factorize(values)
                right.append(codes[:len(source)]);left.append(codes[len(source):])
        left=np.column_stack(left);right=np.column_stack(right)
        def measure(indices):
            equal=left==right[indices];counts=equal.sum(axis=1)
            group_n=np.bincount(ix,minlength=len(source));group_sum=np.bincount(ix,weights=counts,minlength=len(source));present=group_n>0
            return {'equal_income_mean_matching_columns':float(np.mean(group_sum[present]/group_n[present])),'mean_matching_columns':float(counts.mean()),'fraction_at_least_6':float((counts>=6).mean()),'fraction_at_least_8':float((counts>=8).mean()),'all_columns_match':int((counts==len(columns)).sum()),'per_column':dict(zip(columns,equal.mean(axis=0).tolist()))}
        observed=measure(ix);controls={}
        for condition in ['income_bin5000','income_bin5000_city','income_bin5000_city_car']:
            key=(source.Annual_Income_USD//5000).astype(str)
            if 'city' in condition:key=key+'|'+source.City_Type.astype(str)
            if 'car' in condition:key=key+'|'+source.Current_Car_Type.astype(str)
            groups=list(source.groupby(key,sort=False).indices.values());permutations=[]
            for seed in range(10):
                rng=np.random.default_rng(seed);permutation=np.arange(len(source))
                for group in groups:permutation[group]=rng.permutation(group)
                permutations.append(measure(permutation[ix]))
            controls[condition]={'permutations':permutations,'mean_matching_columns_null_mean':float(np.mean([r['mean_matching_columns'] for r in permutations])),'mean_matching_columns_null_max':float(np.max([r['mean_matching_columns'] for r in permutations]))}
        results[split]={'total_rows':len(df),'eligible_rows':int(eligible.sum()),'eligible_fraction':float(eligible.mean()),'observed':observed,'controls':controls}
        print(split,results[split]['eligible_rows'],'observed',observed['mean_matching_columns'],'null',{k:v['mean_matching_columns_null_mean'] for k,v in controls.items()},flush=True)
    result={'unique_income_source_rows':len(source),'columns':columns,'source_sha256':digest(ORIG),'script_sha256':digest(__file__),'results':results,'caveat':'No competition labels loaded. Unique-income rows only; ambiguous source matches excluded. Source blocks permuted within income bins, optionally city, preserving covariate relationships within source blocks. Ten null permutations are diagnostics, not proof of generation or ancestry.'}
    Path('reports/round10/structure.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
