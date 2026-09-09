"""Optional TabM experiment; public references and rationale in reports/round3/report.md."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .expert import ExpertEncoder, TARGET
from .pipeline import digest, score


class NeuralEncoder:
    def fit_transform(self, frame, y, original):
        self.expert = ExpertEncoder('full')
        x = self.expert.fit_transform(frame, y, original)
        # Keep raw/recipe/digit views and exact-value statistics; omit redundant digit TE.
        self.columns = [c for c in x if ('_digit_' not in c and '_frequency' not in c and '_te' not in c)
                        or c in ['Annual_Income_USD_frequency', 'Daily_Commute_km_frequency']
                        or (c.startswith(('Annual_Income_USD_te', 'Daily_Commute_km_te')))
                        or (c.startswith('Annual_Income_USD_digit_') and c.rsplit('_', 1)[-1] in ['0','1','2','3'])]
        self.columns = [c for c in self.columns if x[c].nunique() > 1]
        self.scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                          subsample=100000, random_state=42)
        return self.scaler.fit_transform(x[self.columns]).astype(np.float32)

    def transform(self, frame):
        return self.scaler.transform(self.expert.transform(frame)[self.columns]).astype(np.float32)


def make_model(n_features, k=8, width=128, bins=None):
    from tabm import TabM
    from rtdl_num_embeddings import LinearReLUEmbeddings, PiecewiseLinearEmbeddings
    return TabM.make(n_num_features=n_features, num_embeddings=(LinearReLUEmbeddings(n_features, 8) if bins is None else
                         PiecewiseLinearEmbeddings(bins, 8, activation=True, version='B')),
                     d_out=1, k=k, d_block=width, n_blocks=2, dropout=.1)


def predict(model, x, batch_size=4096):
    import torch
    model.eval()
    result=[]
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch=torch.as_tensor(x[start:start+batch_size], device='cuda')
            result.append(model(batch).squeeze(-1).sigmoid().mean(1).cpu().numpy())
    return np.concatenate(result)


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--folds', default='0')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--rate', type=float, default=.002)
    p.add_argument('--embedding', choices=['linear','piecewise'], default='linear')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--width', type=int, default=128)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    root=Path.home()/'.cache/kaggle/playground-series-s6e9'
    original_path=Path.home()/'.cache/kaggle/ev-original/EV_Adoption_and_Range_Anxiety_Dataset.csv'
    df=pd.read_csv(root/'train.csv'); test=pd.read_csv(root/'test.csv'); original=pd.read_csv(original_path)
    y=df[TARGET].eq('Yes').to_numpy(dtype=np.float32)
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    splits=list(StratifiedKFold(5, shuffle=True, random_state=42).split(df,y))
    for fold in map(int,a.folds.split(',')):
        if (out/f'fold{fold}.json').exists(): continue
        started=time.monotonic(); torch.manual_seed(42+fold); np.random.seed(42+fold)
        tr,va=splits[fold]; enc=NeuralEncoder()
        xt=enc.fit_transform(df.iloc[tr], y[tr], original)
        xv=enc.transform(df.iloc[va]); xx=enc.transform(test)
        joblib.dump(enc,out/f'encoder{fold}.joblib')
        bins=None
        if a.embedding=='piecewise':
            from rtdl_num_embeddings import compute_bins
            bins=compute_bins(torch.as_tensor(xt),n_bins=48)
        model=make_model(xt.shape[1], a.k, a.width, bins).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=a.rate,weight_decay=.01)
        xgpu=torch.as_tensor(xt,device='cuda'); ygpu=torch.as_tensor(y[tr],device='cuda')
        best=-1.; best_epoch=0; history=[]
        print(f'fold={fold} features={xt.shape[1]} GPU={torch.cuda.get_device_name()}',flush=True)
        for epoch in range(1,a.epochs+1):
            model.train(); order=torch.randperm(len(tr),device='cuda')
            for indices in order.split(2048):
                opt.zero_grad(set_to_none=True)
                logits=model(xgpu[indices]).squeeze(-1)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,ygpu[indices,None].expand_as(logits))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                opt.step()
            pv=predict(model,xv); auc=float(roc_auc_score(y[va],pv))
            history.append({'epoch':epoch,'auc':auc})
            print(f'fold={fold} epoch={epoch} auc={auc:.9f} seconds={time.monotonic()-started:.1f}',flush=True)
            if auc>best:
                best=auc;best_epoch=epoch
                torch.save({'state':model.state_dict(),'n_features':xt.shape[1],'k':a.k,'width':a.width,'bins':bins},out/f'fold{fold}.pt')
            if epoch-best_epoch>=a.patience: break
        checkpoint=torch.load(out/f'fold{fold}.pt',weights_only=True)
        model.load_state_dict(checkpoint['state']); pv=predict(model,xv); pt=predict(model,xx)
        np.savez_compressed(out/f'fold{fold}.npz',indices=va,oof=pv,test=pt)
        result={'fold':fold,**score(y[va],pv),'best_epoch':best_epoch,'history':history,'args':vars(a),
                'seconds':time.monotonic()-started,'features':enc.columns,'source_sha256':digest(__file__),
                'train_sha256':digest(root/'train.csv'),'original_sha256':digest(original_path),
                'torch_version':str(torch.__version__),'early_stopping_on_scored_fold':True}
        (out/f'fold{fold}.json').write_text(json.dumps(result,indent=2)+'\n')
        print('COMPLETE',fold,result['auc'],flush=True)
        del model,xgpu,ygpu,xt,xv,xx,enc,opt
        torch.cuda.empty_cache()

if __name__=='__main__':
    from src.neural import main as entry
    entry()
