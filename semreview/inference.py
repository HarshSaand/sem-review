from __future__ import annotations
from pathlib import Path
import hashlib,time
import numpy as np
import torch
from PIL import Image
from semreview.model import TinyUNet
from semreview.imaging import preprocess,measures,overlay,CROP
ROOT=Path(__file__).resolve().parents[1]

def review_score(prob: np.ndarray,mask: np.ndarray):
    # Ranking heuristic, NOT calibrated probability of an incorrect inspection.
    if not mask.any():return 1.,['No defect segmented: verify the image and acquisition alignment.']
    confidence=float(prob[mask].mean());score=float(1-confidence)
    reasons=[]
    if confidence<.85:reasons.append('Low average model confidence in segmented pixels.')
    if mask.sum()<12:reasons.append('Very small region at the model resolution.')
    if mask.mean()>.3:reasons.append('Unusually large segmented fraction.')
    if reasons:score=max(score,.3)
    return score,reasons

class Predictor:
    def __init__(self,path=None,device='cpu'):
        path=Path(path or ROOT/'artifacts/model.pt')
        if not path.exists():raise FileNotFoundError('Train the model first: python scripts/train.py')
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['crop'] != CROP: raise ValueError('Checkpoint crop does not match preprocessing.')
        self.model=TinyUNet(ckpt['width']).eval().to(device);self.model.load_state_dict(ckpt['state_dict'])
        self.device=device;self.threshold=ckpt['threshold'];self.size=ckpt['model_size']
        self.model_id=hashlib.sha256(path.read_bytes()).hexdigest()[:16];self.epoch=ckpt['epoch']
    def predict(self,image:Image.Image):
        started=time.perf_counter();x,box=preprocess(image,self.size)
        with torch.inference_mode():p=self.model(torch.from_numpy(x)[None,None].to(self.device)).sigmoid()[0,0].cpu().numpy()
        score,reasons=review_score(p,p>self.threshold)
        w,h=image.size;rw,rh=box[2]-box[0],box[3]-box[1]
        roi=np.asarray(Image.fromarray(p).resize((rw,rh),Image.Resampling.BILINEAR))
        full=np.zeros((h,w),bool);full[box[1]:box[3],box[0]:box[2]]=roi>self.threshold
        if not full.any() and not reasons: reasons.append('No region survives full-resolution thresholding.');score=1.
        report=measures(full)
        report.update({'model_id':self.model_id,'model_epoch':self.epoch,'threshold':self.threshold,'crop_box_xyxy':list(box),
            'image_size_wh':[w,h],'evaluated_pixels':rw*rh,'coverage_fraction':float(full.sum()/(rw*rh)),
            'review_score':score,'review_required':bool(reasons),'review_reasons':reasons,
            'confidence_mean_foreground':float(p[p>self.threshold].mean()) if (p>self.threshold).any() else None,
            'inference_ms':(time.perf_counter()-started)*1000,'units':'pixels; no physical calibration',
            'category_prediction':None,'scope':'Research demonstration on Carinthia-S; not validated for production disposition.'})
        return full,report
