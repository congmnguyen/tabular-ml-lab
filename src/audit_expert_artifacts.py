"""Export compact provenance from the actual saved estimators and encoders."""
import inspect
import json
from pathlib import Path
import joblib
from .pipeline import digest


def main():
    report=Path('reports/round2/runs');report.mkdir(exist_ok=True,parents=True)
    for run in sorted(Path('artifacts').glob('r2-*')):
        if not run.is_dir():continue
        if (run/'blend.json').exists():
            (report/f'{run.name}.json').write_text((run/'blend.json').read_text());continue
        folds=[]
        for p in sorted(run.glob('fold*.json')):
            if not p.stem[4:].isdigit():continue
            result=json.loads(p.read_text());path=run/f'{p.stem}.joblib'
            if not path.exists():continue
            b=joblib.load(path)
            result['actual_estimator_params']=b['model'].get_params()
            result['encoder_class']=f"{type(b['encoder']).__module__}.{type(b['encoder']).__name__}"
            result['encoder_source_sha256']=digest(inspect.getfile(type(b['encoder'])))
            result['artifact_sha256']=digest(path)
            folds.append(result)
        info={'run':str(run),'complete':(run/'summary.json').exists(),'folds':folds}
        if info['complete']:
            summary=json.loads((run/'summary.json').read_text())
            info.update({k:v for k,v in summary.items() if k!='folds'})
        (report/f'{run.name}.json').write_text(json.dumps(info,indent=2,default=str)+'\n')
    print('Exported estimator and encoder provenance')
if __name__=='__main__':main()
