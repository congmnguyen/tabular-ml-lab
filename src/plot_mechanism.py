"""Plot conditional diagnostics and paired development-fold improvements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('reports/round4');p=json.loads((root/'conditional-permutation.json').read_text());s=json.loads((root/'selected.json').read_text())
    names=list(dict.fromkeys(r['block'] for r in p['results']))
    data=np.array([[r['auc_drop'] for r in p['results'] if r['block']==name] for name in names])*1000
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    ax[0].bar(range(3),data.mean(1),color='#475569',width=.6)
    ax[0].scatter(np.repeat(range(3),2),data.ravel(),color='#2563eb',s=28,zorder=3)
    ax[0].set(xticks=range(3),xticklabels=['Exact income','Commute','Other covariates'],ylabel='AUC drop (× 10⁻³)',title='Conditional permutation: two seeds')
    changes=np.array(s['delta_by_fold'])*1e5
    ax[1].bar(range(1,6),changes,color='#2563eb',width=.6)
    for i,v in enumerate(changes,1):ax[1].text(i,v+.12,f'{v:.2f}',ha='center')
    ax[1].set(xticks=range(1,6),xlabel='Fold',ylabel='AUC gain over round 3 (× 10⁻⁵)',title='Mechanistic offset ensemble',ylim=(0,max(changes)*1.2))
    for a in ax:
        a.grid(axis='y',alpha=.2);a.set_axisbelow(True);a.spines[['top','right']].set_visible(False)
    fig.suptitle('Approximate conditional diagnostics; CV includes selection and early stopping',fontsize=11,color='#475569')
    fig.savefig(root/'mechanism-validation.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
