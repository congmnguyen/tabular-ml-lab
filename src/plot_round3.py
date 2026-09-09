"""Plot measured OOF gains relative to the previously submitted blend."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('reports/round3');r=json.loads((root/'selected.json').read_text())
    prior=r['selected']['mean_auc']-r['mean_auc_gain']
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for kind,color in [('probability','#64748b'),('rank','#2563eb')]:
        rows=[x for x in r['candidates'] if x['kind']==kind]
        axes[0].plot([x['weights'][2]*100 for x in rows],[(x['mean_auc']-prior)*1e5 for x in rows],'-o',label=kind,color=color)
    axes[0].set(xlabel='TabM weight (%)',ylabel='Mean CV AUC gain (× 10⁻⁵)',title='Small blend grid, five-fold CV')
    axes[0].legend(frameon=False)
    gains=np.array(r['delta_by_fold'])*1e5
    axes[1].bar(np.arange(1,6),gains,color='#2563eb',width=.6)
    for i,value in enumerate(gains,1):axes[1].text(i,value+.12,f'{value:.2f}',ha='center')
    axes[1].set(xlabel='Fold',ylabel='AUC gain (× 10⁻⁵)',title='Selected blend improves all five folds',ylim=(0,max(gains)*1.2),xticks=range(1,6))
    for ax in axes:
        ax.axhline(0,color='#334155',lw=.7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Development estimates: repeated screening and early stopping apply',fontsize=11,color='#475569')
    fig.savefig(root/'blend-validation.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
