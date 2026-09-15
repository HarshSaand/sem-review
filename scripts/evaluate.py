"""Freeze validation choices, then evaluate held-out original/near-duplicate groups."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFont
from semreview.model import TinyUNet
from semreview.baseline import classical
from semreview.imaging import CROP,SIZE,segmentation_metrics,overlay
from semreview.inference import review_score
ROOT=Path(__file__).resolve().parents[1]
def dice(a,b):
    s=a.sum()+b.sum();return float(2*(a&b).sum()/s) if s else 1.
def main():
    torch.set_num_threads(4)
    df=pd.read_csv(ROOT/'artifacts/split_manifest.csv');x=np.load(ROOT/'data/processed/images.npy',mmap_mode='r');y=np.load(ROOT/'data/processed/masks.npy',mmap_mode='r')
    val=np.flatnonzero(df.split=='validation');test=np.flatnonzero(df.split=='test')
    baseline_scores=[]
    for mode in ['local','global']:
        for threshold in [6.,10.,14.,18.,24.]:
            scores=[dice(classical(x[i],threshold,mode),y[i].astype(bool)) for i in val]
            baseline_scores.append({'mode':mode,'threshold':threshold,'validation_dice_model_grid':float(np.mean(scores))})
    best=max(baseline_scores,key=lambda r:r['validation_dice_model_grid'])
    (ROOT/'artifacts/baseline_selection.json').write_text(json.dumps({'selected':best,'candidates':baseline_scores},indent=2))
    print('Selected baseline',best,flush=True)
    ck=torch.load(ROOT/'artifacts/model.pt',map_location='cpu',weights_only=True);model=TinyUNet().eval();model.load_state_dict(ck['state_dict'])
    rows=[];predictions={};latency=[];outputs=ROOT/'artifacts/examples';outputs.mkdir(exist_ok=True)
    for batch_start in range(0,len(test),24):
        inds=test[batch_start:batch_start+24]
        with torch.inference_mode():p=model(torch.from_numpy(np.asarray(x[inds],dtype=np.float32)/255.)[:,None]).sigmoid()[:,0].numpy()
        for i,prob in zip(inds,p):
            r=df.iloc[i];image=Image.open(ROOT/r.image_path).convert('L');gt=Image.open(ROOT/r.mask_path).convert('L')
            box=(CROP,CROP,image.width-CROP,image.height-CROP);dims=(image.width-2*CROP,image.height-2*CROP)
            target=np.asarray(gt.crop(box))>0
            prediction=np.asarray(Image.fromarray(prob).resize(dims,Image.Resampling.BILINEAR))>ck['threshold']
            baseline=np.asarray(Image.fromarray(classical(x[i],best['threshold'],best['mode']).astype(np.uint8)*255).resize(dims,Image.Resampling.NEAREST))>0
            score,reasons=review_score(prob,prob>ck['threshold'])
            row={'filename':r.filename,'label':int(r.label),'group_id':int(r.group_id),'split':'test','review_score':score,'review_required':bool(reasons)}
            row.update({'unet_'+k:v for k,v in segmentation_metrics(prediction,target).items()})
            row.update({'baseline_'+k:v for k,v in segmentation_metrics(baseline,target).items()})
            rows.append(row)
            predictions[r.filename]=(prediction,baseline,target,image)
        print('Evaluated',min(batch_start+24,len(test)),'/',len(test),flush=True)
    result=pd.DataFrame(rows);result.to_csv(ROOT/'artifacts/test_per_image.csv',index=False)
    def summary(frame,prefix):
        metrics=['dice','iou','boundary_f1_2px','area_error_pixels','coverage_error']
        return {m:float(frame[prefix+m].mean()) for m in metrics}
    metrics={'test_images':len(test),'test_groups':int(df.iloc[test].group_id.nunique()),'evaluation_grid':'Native image interior after uniform 32px crop (416x416 for 480x480 sources). Model outputs bilinearly resized from192x192, then thresholded.',
         'unet':summary(result,'unet_'),'classical':summary(result,'baseline_'),
         'equal_group_weighted_unet':summary(result.groupby('group_id').mean(numeric_only=True),'unet_'),
         'equal_group_weighted_classical':summary(result.groupby('group_id').mean(numeric_only=True),'baseline_'),
         'per_class':{},'review_coverage':[],'empty_mask':{},'model_threshold':ck['threshold'],'model_epoch':ck['epoch'],
         'selection':'Both checkpoint/threshold and baseline hyperparameters selected on validation only. Test evaluated once after training.',
         'limitations':['Image-level grouped holdout, not a wafer/lot/factory holdout.','Class6 empty captures result from SEM misalignment; not factory clean controls.','Review score is heuristic, not calibrated uncertainty.','No reliable six-class classifier; no physical length or area calibration.']}
    for label,part in result.groupby('label'):
        metrics['per_class'][str(label)]={'n':len(part),'unet':summary(part,'unet_'),'classical':summary(part,'baseline_')}
    for coverage in [.25,.5,.75,1.]:
        subset=result.sort_values(['review_score','filename']).head(max(1,int(len(result)*coverage)))
        metrics['review_coverage'].append({'retained_fraction':coverage,'retained_images':len(subset),'dice':float(subset.unet_dice.mean()),'iou':float(subset.unet_iou.mean())})
    empty=result[result.unet_target_area==0]
    metrics['empty_mask']={'n':len(empty),'unet_nonempty_predictions':int((empty.unet_predicted_area>0).sum()),'baseline_nonempty_predictions':int((empty.baseline_predicted_area>0).sum())}
    # Warmed serial CPU timing, isolated from batches and disk I/O; supplement API end-to-end timing.
    inp=torch.from_numpy(np.asarray(x[test[0]],dtype=np.float32)/255.)[None,None]
    with torch.inference_mode():
        for _ in range(3):model(inp)
        for _ in range(30):
            s=time.perf_counter();model(inp);latency.append((time.perf_counter()-s)*1000)
    metrics['cpu_forward_latency_ms']={'median':float(np.median(latency)),'p95':float(np.percentile(latency,95)),'n':30,'threads':4,'includes_preprocessing':False}
    metrics['model_sha256']=hashlib.sha256((ROOT/'artifacts/model.pt').read_bytes()).hexdigest()
    (ROOT/'artifacts/metrics.json').write_text(json.dumps(metrics,indent=2))
    # Deterministic representative outputs: median-scoring example per supported class plus low-Dice failure.
    selected=[]
    for label,part in result.groupby('label'):
        part=part.sort_values('unet_dice');selected.append((str(label),part.iloc[len(part)//2].filename))
    failure=result[result.unet_target_area>0].sort_values('unet_dice').iloc[0].filename
    if failure not in [v for _,v in selected]:selected.append(('failure',failure))
    cards=[];index=[]
    try:font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
    except OSError:font=ImageFont.load_default()
    for label,name in selected:
        pred,base,truth,image=predictions[name];folder=outputs/name;folder.mkdir(exist_ok=True)
        full=np.zeros((image.height,image.width),bool);full[CROP:-CROP,CROP:-CROP]=pred
        fg=np.zeros_like(full);fg[CROP:-CROP,CROP:-CROP]=truth
        fb=np.zeros_like(full);fb[CROP:-CROP,CROP:-CROP]=base
        image.save(folder/'input.png');Image.fromarray(full.astype('uint8')*255).save(folder/'prediction.png')
        Image.fromarray(fg.astype('uint8')*255).save(folder/'ground_truth.png');Image.fromarray(fb.astype('uint8')*255).save(folder/'baseline.png')
        overlay(image,full).save(folder/'overlay.png')
        row=result[result.filename==name].iloc[0].to_dict();(folder/'metrics.json').write_text(json.dumps(row,indent=2))
        tile=Image.new('RGB',(4*300,350),'#111d26');draw=ImageDraw.Draw(tile)
        for col,(im,title) in enumerate([(image,'Held-out input'),(overlay(image,fg),'Expert ground truth'),(overlay(image,full),f'U-Net Dice {row["unet_dice"]:.3f}'),(overlay(image,fb),f'Classical Dice {row["baseline_dice"]:.3f}')]):
            tile.paste(im.convert('RGB').resize((300,300)),(col*300,40));draw.text((col*300+12,10),title,font=font,fill='white')
        tile.save(folder/'panel.png');cards.append(tile)
        index.append({'filename':name,'label':int(row['label']),'selection':'lowest foreground Dice' if label=='failure' else 'median Dice within held-out class','dice':row['unet_dice'],'directory':f'artifacts/examples/{name}'})
    allpanels=Image.new('RGB',(1200,len(cards)*350))
    for i,tile in enumerate(cards):allpanels.paste(tile,(0,i*350))
    allpanels.save(ROOT/'artifacts/heldout_panels.png')
    (ROOT/'artifacts/examples/index.json').write_text(json.dumps(index,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
