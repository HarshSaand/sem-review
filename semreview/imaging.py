from __future__ import annotations
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CROP=32
SIZE=192

def crop_box(size: tuple[int,int], border: int=CROP):
    w,h=size
    if min(w,h)<128: raise ValueError('Image must be at least 128 × 128 pixels.')
    if max(w,h)>4096: raise ValueError('Image dimensions must not exceed 4096 pixels.')
    return (border,border,w-border,h-border)

def preprocess(image: Image.Image, size: int=SIZE):
    box=crop_box(image.size)
    roi=image.convert('L').crop(box)
    small=np.asarray(roi.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255.
    return small,box

def measures(mask: np.ndarray):
    labels,n=ndi.label(mask)
    objects=ndi.find_objects(labels)
    regions=[]
    for i,s in enumerate(objects,1):
        if s is None: continue
        region=(labels[s]==i); area=int(region.sum())
        yy,xx=np.where(region)
        regions.append({'region_id':i,'area_pixels':area,'bbox_xyxy':[s[1].start,s[0].start,s[1].stop,s[0].stop],
            'centroid_xy':[float(xx.mean()+s[1].start),float(yy.mean()+s[0].start)]})
    return {'area_pixels':int(mask.sum()),'coverage_fraction':float(mask.mean()),'region_count':n,'regions':regions}

def overlay(image: Image.Image, mask: np.ndarray):
    a=np.asarray(image.convert('RGB')).copy()
    a[mask]=(a[mask]*.45+np.array([255,92,47])*.55).astype(np.uint8)
    return Image.fromarray(a)

def segmentation_metrics(pred: np.ndarray, target: np.ndarray):
    pred=pred.astype(bool); target=target.astype(bool)
    tp=int((pred&target).sum()); fp=int((pred&~target).sum()); fn=int((~pred&target).sum())
    dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    iou=tp/(tp+fp+fn) if tp+fp+fn else 1.
    # Boundary F1 with 2 pixels of tolerance, on the specified evaluation grid.
    pb=pred^ndi.binary_erosion(pred); tb=target^ndi.binary_erosion(target)
    if not pb.any() and not tb.any(): bf=1.
    elif not pb.any() or not tb.any(): bf=0.
    else:
        precision=(pb&ndi.binary_dilation(tb,iterations=2)).sum()/pb.sum()
        recall=(tb&ndi.binary_dilation(pb,iterations=2)).sum()/tb.sum()
        bf=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'dice':float(dice),'iou':float(iou),'boundary_f1_2px':float(bf),
            'area_error_pixels':int(abs(int(pred.sum())-int(target.sum()))),
            'coverage_error':float(abs(pred.mean()-target.mean())),
            'target_area':int(target.sum()),'predicted_area':int(pred.sum())}
