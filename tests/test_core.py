from pathlib import Path
import io,json,zipfile
import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient
from semreview.imaging import crop_box,measures,segmentation_metrics,CROP
from semreview.model import TinyUNet

def test_empty_and_disjoint_metrics():
    z=np.zeros((16,16),bool);one=z.copy();one[2:5,2:5]=True
    assert segmentation_metrics(z,z)['dice']==1
    assert segmentation_metrics(one,z)['dice']==0
    assert segmentation_metrics(one,one)['boundary_f1_2px']==1

def test_pixel_regions_are_measured_not_physical():
    a=np.zeros((20,20),bool);a[2:5,3:7]=True;a[10,10]=True
    m=measures(a);assert m['area_pixels']==13 and m['region_count']==2
    assert m['regions'][0]['bbox_xyxy']==[3,2,7,5]
    assert crop_box((480,480))==(32,32,448,448)
    with pytest.raises(ValueError):crop_box((60,60))

def test_unet_restores_image_extent():
    torch.set_num_threads(2)
    with torch.inference_mode():out=TinyUNet()(torch.zeros(1,1,64,64))
    assert out.shape==(1,1,64,64) and torch.isfinite(out).all()

def test_group_manifest_has_no_leakage():
    import pandas as pd
    path=Path(__file__).resolve().parents[1]/'artifacts/split_manifest.csv'
    if not path.exists():pytest.skip('Run prepare_data.py first')
    df=pd.read_csv(path)
    assert df.groupby('group_id').split.nunique().max()==1
    assert df.groupby('sha256_cropped_image').split.nunique().max()==1
    assert len(df)==4591 and set(df.split)=={'train','validation','test'}

def png(array):
    b=io.BytesIO();Image.fromarray(array).save(b,format='PNG');return b.getvalue()

def test_review_correction_and_export_preserve_prediction(tmp_path,monkeypatch):
    import semreview.app as appmod
    class Stub:
        model_id='test'
        def predict(self,image):
            a=np.zeros((image.height,image.width),bool);a[50:60,50:60]=True
            r=measures(a);r.update({'model_id':'test','image_size_wh':list(image.size),'crop_box_xyxy':[32,32,128,128],
              'evaluated_pixels':96*96,'review_required':False,'review_reasons':[],'units':'pixels'})
            return a,r
    monkeypatch.setattr(appmod,'Predictor',Stub);monkeypatch.setattr(appmod,'REVIEWS',tmp_path)
    with TestClient(appmod.app) as client:
        response=client.post('/api/inspect',files={'file':('sample.png',png(np.full((160,160),120,dtype=np.uint8)),'image/png')})
        assert response.status_code==200;record=response.json();key=record['review_id']
        correction=np.zeros((160,160),np.uint8);correction[60:65,70:76]=255;correction[:8]=255
        r=client.post(f'/api/reviews/{key}/correct',files={'mask':('mask.png',png(correction),'image/png')})
        assert r.status_code==200;corrected=r.json()
        assert corrected['area_pixels']==30 and corrected['status']=='corrected' and corrected['changed_pixels_from_model']==130
        assert corrected['coverage_fraction']==30/(96*96)
        archive=zipfile.ZipFile(io.BytesIO(client.get(f'/api/reviews/{key}/export').content))
        assert {'prediction.png','mask.png','report.json','regions.csv','inspection.csv'}.issubset(archive.namelist())
        assert np.asarray(Image.open(io.BytesIO(archive.read('prediction.png')))).astype(bool).sum()==100
        assert np.asarray(Image.open(io.BytesIO(archive.read('mask.png')))).astype(bool).sum()==30
        assert client.post(f'/api/reviews/{key}/accept').json()['status']=='accepted'
        assert client.post(f'/api/reviews/{key}/correct',files={'mask':('bad.png',png(np.zeros((128,128),np.uint8)))}).status_code==400
        assert client.post('/api/inspect',files={'file':('bad.png',b'not an image')}).status_code==400
