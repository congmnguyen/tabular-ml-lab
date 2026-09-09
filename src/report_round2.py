"""Summarize actual round-two experiments without converting screens into full CV claims."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submissions',default='reports/round2/submissions.json');args=parser.parse_args()
    out=Path('reports/round2');runs=out/'runs'
    models=[json.loads(p.read_text()) for p in sorted(runs.glob('*.json')) if 'blend' not in p.name]
    rows=['# Round 2 — expert-informed data investigation and leaderboard attempt','','Target: exceed Chris Deotte\'s observed public AUC of **0.94672**. This round extends the initial five-model case study.','','## What changed after studying experts','','See [research and attribution](research.md) for public sources, hypotheses and independent checks. The major gain came from representing repeated synthetic values with original-data statistics, frequency features and internally cross-fitted target encodings. The public Chris Deotte notebooks are starters, not a disclosed reproduction of his leading submission.','','![Data mechanisms](data-mechanism.png)','','## Fold-0 screens','','All screens use the same outer fold. These are development comparisons, not full cross-validation results.','','| Run | Fold-0 AUC | Best iteration | Expanded to five folds |','|---|---:|---:|---|']
    for m in models:
        if not m.get('folds'):continue
        f=m['folds'][0]
        rows.append(f"| {Path(m['run']).name} | {f['auc']:.6f} | {f['best_iteration']} | {'Yes' if m['complete'] else 'No'} |")
    rows+=['','## Completed five-fold models','','| Run | Mean fold AUC | OOF AUC | OOF log loss |','|---|---:|---:|---:|']
    for m in models:
        if m.get('complete'):rows.append(f"| {Path(m['run']).name} | {m['mean_auc']:.6f} | {m['oof']['auc']:.6f} | {m['oof']['log_loss']:.6f} |")
    rows+=['','## Blend selection','','Only models with complete, aligned OOF predictions are eligible. A small grid of fixed weights is compared on mean fold AUC. Rank outputs are ranking scores, not calibrated probabilities. Detailed candidate grids and correlations are saved in [run records](runs/).','','## Kaggle results','','| Submission | Description | Public AUC | Status |','|---|---|---:|---|']
    source=Path(args.submissions)
    submissions=json.loads(source.read_text()) if source.suffix=='.json' else list(csv.DictReader(source.open()))
    for s in submissions:rows.append(f"| {s['ref']} | {s['description']} | {s['publicScore']} | {s['status']} |")
    scored=[float(s['publicScore']) for s in submissions if s['publicScore']]
    if scored:
        best=max(scored);rows+=['',f'Best observed public AUC: **{best:.5f}**. Gap to the requested 0.94672 target: **{0.94672-best:+.5f}**. '+('Target exceeded.' if best>.94672 else 'The target has not been exceeded.'),'']
    rows+=['## Interpretation and limits','','- The original recipe explains broad behavior; original-value repetition and synthetic-generator distortions explain why the initial generic pipeline left substantial signal unused.',
           '- Two full-feature tree models have OOF correlation around 0.99932, limiting ensemble gains. Small screening changes do not establish robust improvement.',
           '- Residual/replication features failed their initial screen. Native CatBoost was weaker alone; its 25% blend gain on fold 0 was tiny. Negative results are retained.',
           '- Fold 0 was reused for screening. Outer-fold early stopping also uses that fold\'s labels. These are development OOF scores, with selection optimism; they are not an independent final generalization estimate.',
           '- Transductive candidates, if present, use train+test feature frequencies only. Test targets are never available. Supervised encoders remain confined to outer training labels with inner cross-fitting.',
           '- Original data are public CC0 data. Raw data, reference notebooks, predictions and model artifacts are not committed. Public sources are credited; no external prediction files were submitted or averaged.',
           '- Final private leaderboard results remain unavailable. GPU CatBoost can vary across runs. No final rank or medal is claimed.', '',
           '## Reproduce','','Install the pinned requirements and download the public original dataset using the command in the README. Examples:', '', '```bash',
           'python -m src.investigate','python -m src.plot_investigation',
           'python -m src.expert --output artifacts/new-full-lgb --model lgb --mode full --rounds 6000',
           'python -m src.expert --output artifacts/new-full-xgb --model xgb --mode full --rounds 5000',
           'python -m src.blend_experts --runs artifacts/new-full-lgb artifacts/new-full-xgb --output artifacts/new-blend',
           'python -m src.predict_expert --run artifacts/new-blend --test data/ev-purchases/test.csv --output artifacts/reproduced.csv',
           '```','','Per-run records include actual saved estimator parameters and encoder classes; these override the shared runner\'s original parameter dictionary for fine-bin variants. Original-source and training-data hashes are recorded for complete runs.','']
    (out/'report.md').write_text('\n'.join(rows))
    (out/'submissions.json').write_text(json.dumps(submissions,indent=2)+'\n')
    print('Updated round-two report')
if __name__=='__main__':main()
