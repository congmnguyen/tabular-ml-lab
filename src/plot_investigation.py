"""Visualize conditional relationships and synthetic value replication."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .investigate import DATA, ORIG

def main():
    t=pd.read_csv(DATA/'train.csv');o=pd.read_csv(ORIG)
    y=t.Will_Buy_EV.eq('Yes')
    out=Path('reports/round2')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for df,name in [(o,'Original'),(t,'Competition')]:
        axes[0].hist(df.Annual_Income_USD.dropna(),bins=np.arange(25000,200001,2500),weights=np.ones(df.Annual_Income_USD.notna().sum())/df.Annual_Income_USD.notna().sum(),histtype='step',label=name)
    axes[0].set(xlabel='Annual income (USD)',ylabel='Fraction of rows',title='Income distribution and 30k spike');axes[0].legend()
    result={}
    for col in ['Annual_Income_USD','Daily_Commute_km']:
        counts=o[col].value_counts(normalize=True)
        ratio=t[col].map(t[col].value_counts(normalize=True))/t[col].map(counts)
        result[col]={'original_value_match_rate':float(t[col].isin(o[col].dropna()).mean()),'share_at_least_double_original_frequency':float(ratio.ge(2).mean())}
    for home,style in [('All','o-'),('Yes','s-'),('No','^-')]:
        rates=[]
        for low,high in [(0,2),(11,14)]:
            mask=t.Charging_Stations_Near_Home.between(low,high)
            if home!='All':mask &= t.Home_Charging_Possible.eq(home)
            rates.append(y[mask].mean())
        axes[1].plot([0,1],rates,style,label=f'Home charging: {home}')
    axes[1].set_xticks([0,1],['0–2 stations','11–14 stations']);axes[1].set(ylabel='Purchase rate',title='Conditioning reverses the aggregate trend');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'data-mechanism.png',dpi=170)
    (out/'value-replication.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
