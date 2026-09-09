import json
import numpy as np
import pandas as pd
from src import blend_experts


def test_blend_scores_all_ten_folds(tmp_path,monkeypatch):
    runs=[]
    for name,variable in [('variable',True),('uninformative',False)]:
        run=tmp_path/name;run.mkdir();runs.append(str(run))
        target=np.tile([0,1,0,1],10)
        prediction=np.r_[np.tile([.1,.9,.2,.8],5),np.tile([.9,.1,.8,.2],5)] if variable else np.tile([.1,.2,.2,.1],10)
        pd.DataFrame({'id':np.arange(40),'target':target,'fold':np.repeat(np.arange(10),4),'prediction':prediction}).to_csv(run/'oof.csv',index=False)
        pd.DataFrame({'id':[50,51],'Will_Buy_EV':[.2,.8]}).to_csv(run/'submission.csv',index=False)
    output=tmp_path/'blend'
    monkeypatch.setattr('sys.argv',['blend_experts','--runs',*runs,'--output',str(output)])
    blend_experts.main()
    result=json.loads((output/'blend.json').read_text())
    assert result['candidates'][0]['mean_auc']==.5
    assert len(result['candidates'][0]['fold_auc'])==10
