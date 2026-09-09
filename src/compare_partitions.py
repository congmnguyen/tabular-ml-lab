"""Compare OOF predictions across different training partitions by aligned IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True)
    p.add_argument('--references', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    candidate = pd.read_csv(Path(a.candidate)/'oof.csv')
    assert candidate.id.is_unique
    output = {'candidate': a.candidate, 'candidate_pooled_auc': float(roc_auc_score(candidate.target, candidate.prediction)), 'comparisons': []}
    for run in a.references:
        reference = pd.read_csv(Path(run)/'oof.csv')
        assert reference.id.is_unique
        assert np.array_equal(candidate[['id','target']], reference[['id','target']])
        reference_auc = float(roc_auc_score(reference.target, reference.prediction))
        groups = []
        for f in sorted(reference.fold.unique()):
            mask = reference.fold == f
            old = float(roc_auc_score(reference.target[mask], reference.prediction[mask]))
            new = float(roc_auc_score(candidate.target[mask], candidate.prediction[mask]))
            groups.append({'reference_fold': int(f), 'reference_auc': old, 'candidate_auc': new, 'difference': new-old})
        output['comparisons'].append({'reference': run, 'reference_pooled_auc': reference_auc, 'pooled_difference': output['candidate_pooled_auc']-reference_auc, 'same_row_groups': groups})
    output['caveat'] = 'Same-row groups are descriptive comparisons, not paired training folds or independent statistical replicates. No cross-partition blending.'
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
