"""Recover missing public-source covariates from its verified RNG sequence.
Algorithm reference: yhay81/exact-reconstruction-of-the-ev-source-dataset.
Competition labels are never inferred or modified.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .investigate import ORIG,DATA
from .pipeline import digest


def main():
    source=pd.read_csv(ORIG);n=len(source);rng=np.random.RandomState(101);draw={}
    draw['Age']=rng.randint(25,70,n)
    draw['Gender']=rng.choice(['Male','Female','Other'],n,p=[.52,.45,.03])
    draw['Annual_Income_USD']=np.maximum(rng.normal(85000,35000,n),30000).astype(int)
    draw['City_Type']=rng.choice(['Urban','Suburban','Rural'],n,p=[.5,.35,.15])
    draw['Number_of_Cars_Owned']=rng.choice([1,2,3,4],n,p=[.4,.4,.15,.05])
    draw['Current_Car_Type']=rng.choice(['Sedan','SUV','Hatchback','Truck'],n,p=[.4,.35,.15,.1])
    draw['Daily_Commute_km']=np.maximum(rng.normal(40,25,n),5).round(1)
    home=[];work=[];possible=[]
    rules={'Urban':(2,15,3,20,.4),'Suburban':(0,8,1,10,.8),'Rural':(0,3,0,4,.9)}
    for city in draw['City_Type']:
        a,b,c,d,p=rules[city];home.append(rng.randint(a,b));work.append(rng.randint(c,d));possible.append('Yes' if rng.rand()<p else 'No')
    draw['Charging_Stations_Near_Home']=home;draw['Charging_Stations_Near_Work']=work;draw['Home_Charging_Possible']=possible
    draw['Environmental_Concern_Level']=rng.randint(1,6,n)
    draw['Subsidy_Available']=rng.choice(['Yes','No'],n,p=[.6,.4])
    matches={}
    for col,values in draw.items():
        known=source[col].notna();equal=source.loc[known,col].to_numpy()==np.asarray(values)[known]
        matches[col]={'observed_cells':int(known.sum()),'mismatches':int((~equal).sum())}
        assert equal.all(),f'RNG reconstruction does not match {col}'
    repaired=source.copy();filled={}
    for col in ['Annual_Income_USD','Daily_Commute_km','Environmental_Concern_Level']:
        missing=source[col].isna();repaired.loc[missing,col]=np.asarray(draw[col])[missing];filled[col]=int(missing.sum())
    coverage={}
    for split in ['train','test']:
        df=pd.read_csv(DATA/f'{split}.csv');coverage[split]={}
        for col in ['Annual_Income_USD','Daily_Commute_km']:
            before=df[col].isin(source[col].dropna());after=df[col].isin(repaired[col]);coverage[split][col]={'before':int(before.sum()),'after':int(after.sum()),'newly_covered':int((after&~before).sum())}
    output=ORIG.parent/'recovered-features.csv';repaired.to_csv(output,index=False)
    result={'reference':'https://www.kaggle.com/code/yhay81/exact-reconstruction-of-the-ev-source-dataset','observed_matches':matches,'filled':filled,'coverage':coverage,
            'original_sha256':digest(ORIG),'repaired_sha256':digest(output),'labels_unchanged':source.Will_Buy_EV.equals(repaired.Will_Buy_EV)}
    Path('reports/round6/source-recovery.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
