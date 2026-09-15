"""Train on training groups; select checkpoint and probability threshold on validation only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from semreview.model import TinyUNet
from semreview.imaging import SIZE,CROP
ROOT=Path(__file__).resolve().parents[1]
class Pairs(Dataset):
    def __init__(self,x,y,idx,augment=False):self.x=x;self.y=y;self.idx=idx;self.augment=augment
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        k=self.idx[i];x=self.x[k].astype(np.float32)/255.;y=self.y[k].astype(np.float32)
        if self.augment:
            turns=np.random.randint(4);x=np.rot90(x,turns);y=np.rot90(y,turns)
            if np.random.rand()<.5:x=np.fliplr(x);y=np.fliplr(y)
            if np.random.rand()<.5:x=np.flipud(x);y=np.flipud(y)
            x=np.clip(x*np.random.uniform(.9,1.1)+np.random.uniform(-.04,.04),0,1)
        return torch.from_numpy(x.copy())[None],torch.from_numpy(y.copy())[None]

def loss(logits,y):
    p=logits.sigmoid();axes=(1,2,3)
    dice=1-((2*(p*y).sum(axes)+1)/(p.sum(axes)+y.sum(axes)+1)).mean()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+dice

def score(prob,y,t):
    pred=prob>t; axes=(1,2,3);tp=(pred*y).sum(axes);s=pred.sum(axes)+y.sum(axes)
    return float(np.where(s>0,2*tp/np.maximum(s,1),1).mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=24);a=ap.parse_args()
    torch.manual_seed(41);np.random.seed(41);random.seed(41);torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    df=pd.read_csv(ROOT/'artifacts/split_manifest.csv');x=np.load(ROOT/'data/processed/images.npy',mmap_mode='r');y=np.load(ROOT/'data/processed/masks.npy',mmap_mode='r')
    tr=np.flatnonzero(df.split=='train');va=np.flatnonzero(df.split=='validation')
    # Square-root inverse frequency prevents majority domination without pretending rare categories are reliable.
    counts=df.iloc[tr].label.value_counts();weights=np.array([1/np.sqrt(counts[v]) for v in df.iloc[tr].label])
    sampler=WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(tr),replacement=True,generator=torch.Generator().manual_seed(41))
    train=DataLoader(Pairs(x,y,tr,True),batch_size=a.batch_size,sampler=sampler,num_workers=0)
    valid=DataLoader(Pairs(x,y,va),batch_size=a.batch_size,num_workers=0)
    model=TinyUNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    best=-1;history=[];start=time.time()
    print('Training',len(tr),'validation',len(va),'device',device,'params',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(a.epochs):
        began=time.time();model.train();losses=[]
        for xb,yb in train:
            xb=xb.to(device);yb=yb.to(device);opt.zero_grad();ls=loss(model(xb),yb);ls.backward();opt.step();losses.append(float(ls.detach().cpu()))
        model.eval();ps=[];ys=[]
        with torch.inference_mode():
            for xb,yb in valid:ps.append(model(xb.to(device)).sigmoid().cpu().numpy());ys.append(yb.numpy())
        p=np.concatenate(ps);target=np.concatenate(ys)
        thresholds=[.25,.35,.45,.5,.55,.65,.75];scores=[score(p,target,t) for t in thresholds]
        j=int(np.argmax(scores));sc=scores[j];threshold=thresholds[j]
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_dice':sc,'threshold':threshold,'seconds':time.time()-began};history.append(record)
        print(json.dumps(record),flush=True)
        if sc>best:
            best=sc
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'threshold':threshold,'epoch':epoch+1,'width':12,'model_size':SIZE,'crop':CROP,'seed':41,'validation_dice':sc},ROOT/'artifacts/model.pt')
        (ROOT/'artifacts/training_history.json').write_text(json.dumps(history,indent=2));schedule.step()
    config={'device':device,'epochs':a.epochs,'batch_size':a.batch_size,'seed':41,'seconds':time.time()-start,'parameters':sum(p.numel() for p in model.parameters()),'model':'TinyUNet width12 GroupNorm','loss':'BCE plus soft Dice','sampling':'inverse square root of training label frequency','selection':'highest validation per-image Dice across fixed thresholds; test untouched'}
    (ROOT/'artifacts/training_config.json').write_text(json.dumps(config,indent=2))
if __name__=='__main__':main()
