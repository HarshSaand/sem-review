"""Classical contrast / morphology baseline; parameters selected on validation."""
import numpy as np
from scipy import ndimage as ndi

def classical(image, threshold=12., mode='local'):
    x=image.astype(float)
    smooth=ndi.gaussian_filter(x,.8)
    background=ndi.gaussian_filter(smooth,16) if mode=='local' else np.median(smooth)
    mask=np.abs(smooth-background)>threshold
    mask=ndi.binary_closing(mask,iterations=2)
    mask=ndi.binary_fill_holes(mask)
    labels,n=ndi.label(mask)
    sizes=np.bincount(labels.ravel());keep=sizes>=8;keep[0]=False
    return keep[labels]
