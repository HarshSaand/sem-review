"""Validate pairs, remove fixed acquisition borders, group near duplicates before split."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,collections,warnings
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedGroupKFold
from semreview.imaging import CROP,SIZE,crop_box
ROOT=Path(__file__).resolve().parents[1]
def main():
    raw=ROOT/'data/raw';out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    candidates=list(raw.rglob('*.csv'))
    candidates=[p for p in candidates if 'image_path' in p.read_text()[:200]]
    if len(candidates)!=1:raise RuntimeError(f'Expected one dataset CSV, got {candidates}')
    csv=candidates[0];df=pd.read_csv(csv,sep=';');base=csv.parent
    images=[];masks=[];hashes=[];thumbs=[];central=[];shas=[];bad=[];outside=[];native_shapes=[]
    for index,row in df.iterrows():
        ip=base/row.image_path;mp=base/row.mask_path
        image=Image.open(ip).convert('L');mask=Image.open(mp).convert('L')
        if image.size!=mask.size:raise ValueError(f'Mismatched shape: {row.filename}')
        original=np.array(mask)>0;box=crop_box(image.size)
        im=image.crop(box);ma=mask.crop(box)
        if not set(np.unique(np.array(mask))).issubset({0,1,255}):bad.append(row.filename)
        a=np.array(im.resize((SIZE,SIZE),Image.Resampling.BILINEAR),dtype=np.uint8)
        m=np.array(ma.resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
        images.append(a);masks.append(m.astype(np.uint8));native_shapes.append(list(image.size))
        outside.append(int(original.sum()-(np.array(ma)>0).sum()))
        t=np.asarray(im.resize((32,32),Image.Resampling.BILINEAR),dtype=float)
        coef=dctn(t,norm='ortho')[:8,:8].ravel()[1:]
        hashes.append(coef>np.median(coef));thumbs.append(np.asarray(im.resize((128,128),Image.Resampling.BILINEAR),dtype=np.float32));central.append(np.asarray(im.crop((im.width//2-64,im.height//2-64,im.width//2+64,im.height//2+64)),dtype=np.float32).ravel());shas.append(hashlib.sha256(np.asarray(im).tobytes()).hexdigest())
        if index%1000==0:print('Prepared',index,flush=True)
    hashes=np.array(hashes);thumbs=np.array(thumbs);central=np.array(central);central=central-central.mean(axis=1,keepdims=True);central=central/np.maximum(np.linalg.norm(central,axis=1,keepdims=True),1e-8);parent=np.arange(len(df))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    def union(i,j):
        a,b=root(i),root(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    exact={}
    for i,h in enumerate(shas):
        if h in exact:union(i,exact[h])
        else:exact[h]=i
    # Frozen before final training/test evaluation; decided from image audit: pHash<=4, 128px RMSE<=2 gray levels, central128 correlation>=.995.
    # Earlier32px RMSE8 rule merged morphology classes; see initial_grouping_diagnostic.
    nn=NearestNeighbors(radius=4/63,metric='hamming',algorithm='brute',n_jobs=2).fit(hashes)
    neighbors=nn.radius_neighbors(hashes,return_distance=False)
    near=[]
    for i,js in enumerate(neighbors):
        for j in js:
            if j<=i:continue
            error=float(np.sqrt(np.mean((thumbs[i]-thumbs[j])**2)))
            if error<=2 and float(central[i] @ central[j])>=.995:union(i,int(j));near.append([i,int(j),error])
    groups=np.array([root(i) for i in range(len(df))]);folds=np.zeros(len(df),int)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=41)
    for k,(_,test) in enumerate(splitter.split(df,df.label,groups)):folds[test]=k
    splits=np.where(folds==0,'test',np.where(folds==1,'validation','train'))
    df['group_id']=groups;df['split']=splits;df['sha256_cropped_image']=shas
    df['image_path']=[str((base/p).relative_to(ROOT)) for p in df.image_path]
    df['mask_path']=[str((base/p).relative_to(ROOT)) for p in df.mask_path]
    df['foreground_pixels_model_grid']=np.array(masks).sum(axis=(1,2));df['foreground_pixels_removed_by_crop']=outside
    df.to_csv(ROOT/'artifacts/split_manifest.csv',index=False)
    np.save(out/'images.npy',np.array(images));np.save(out/'masks.npy',np.array(masks))
    audit={'dataset':'Carinthia-S','doi':'10.5281/zenodo.16895427','license':'CC-BY-4.0','images':len(df),
       'image_sizes':dict(collections.Counter(map(str,native_shapes))),'labels':{str(k):int(v) for k,v in df.label.value_counts().sort_index().items()},
       'split_counts':df.split.value_counts().to_dict(),'split_class_counts':pd.crosstab(df.split,df.label).to_dict(),
       'exact_duplicate_count':len(df)-len(exact),'near_duplicate_edges':len(near),'groups':len(np.unique(groups)),
       'largest_group':int(pd.Series(groups).value_counts().max()),'grouping':'SHA256 cropped pixels OR pHash Hamming<=4/63 AND 128x128 RMSE<=2 gray levels AND central128 correlation>=0.995; connected components; frozen before final training and test evaluation',
       'split':'StratifiedGroupKFold 5 folds seed41; fold0 test, fold1 validation, others train; groups never overlap',
       'border_crop_each_side_pixels':CROP,'model_size':SIZE,'nonbinary_mask_count':len(bad),'nonbinary_masks':bad,'mask_binarisation':'Any nonzero annotation pixel is foreground, consistently at training and native evaluation. This retains anti-aliased edge pixels.',
       'masks_with_cropped_foreground':int(np.count_nonzero(outside)),'foreground_pixels_removed':int(sum(outside)),
       'empty_masks_model_grid':int((df.foreground_pixels_model_grid==0).sum()),
       'limitations':['No wafer/lot identifiers: grouping cannot establish fab-level generalisation.','Rare classes have insufficient support for six-class classification.','Class 6 misaligned captures are not representative clean wafers.','Centre placement and SEM framing artifacts remain possible shortcuts.']}
    (ROOT/'artifacts/data_audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'artifacts/near_duplicate_pairs.json').write_text(json.dumps(near))
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
