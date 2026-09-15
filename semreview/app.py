"""Local inspection service. Model annotations require human acceptance."""
from pathlib import Path
from datetime import datetime,timezone
from contextlib import asynccontextmanager
import io,json,uuid,re,zipfile,csv,threading,hashlib
import numpy as np
import torch
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from semreview.inference import Predictor
from semreview.imaging import overlay,measures
ROOT=Path(__file__).resolve().parents[1];REVIEWS=ROOT/'reviews';LOCK=threading.Lock()
predictor=None
@asynccontextmanager
async def lifespan(app):
    global predictor
    torch.set_num_threads(4);predictor=Predictor();REVIEWS.mkdir(exist_ok=True)
    yield
app=FastAPI(title='SEM Review',version='0.1.0',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
@app.get('/')
def home():return FileResponse(ROOT/'static/index.html')
@app.get('/api/health')
def health():return {'ready':predictor is not None,'model_id':predictor.model_id if predictor else None}
@app.get('/api/evidence')
def evidence():
    files=['data_audit.json','metrics.json','training_config.json']
    return {f.removesuffix('.json'):json.loads((ROOT/'artifacts'/f).read_text()) for f in files if (ROOT/'artifacts'/f).exists()}
@app.get('/api/examples')
def examples():
    index=ROOT/'artifacts/examples/index.json'
    return json.loads(index.read_text()) if index.exists() else []
@app.get('/api/examples/{name}/{kind}')
def example(name:str,kind:str):
    if not re.fullmatch('[a-f0-9]{32}',name) or kind not in ['input','prediction','ground_truth','overlay','panel']:raise HTTPException(404)
    p=ROOT/'artifacts/examples'/name/f'{kind}.png'
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p)

def folder(key):
    if not re.fullmatch('[a-f0-9]{32}',key):raise HTTPException(404)
    p=REVIEWS/key
    if not p.is_dir():raise HTTPException(404)
    return p

def public_report(p):
    report=json.loads((p/'report.json').read_text());key=p.name
    report['artifacts']={name:f'/api/reviews/{key}/files/{name}' for name in ['input.png','prediction.png','mask.png','overlay.png','report.json','regions.csv']}
    report['export_url']=f'/api/reviews/{key}/export'
    return report

def save_report(p,report):
    (p/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (p/'regions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['region_id','area_pixels','bbox_xyxy','centroid_xy'])
        writer.writeheader();writer.writerows(report['regions'])
    with (p/'inspection.csv').open('w',newline='') as f:
        keys=['review_id','filename','status','area_pixels','coverage_fraction','region_count','review_required','model_id','units']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerow({k:report[k] for k in keys})

def read_image(body):
    if len(body)>12*1024*1024:raise HTTPException(413,'Maximum file size is12MB.')
    try:
        im=Image.open(io.BytesIO(body));im.load();return im.convert('L')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):raise HTTPException(400,'Upload a valid PNG, JPEG or TIFF image.')

@app.post('/api/inspect')
async def inspect(file:UploadFile=File(...)):
    global predictor
    image=read_image(await file.read(12*1024*1024+1))
    try:
        with LOCK:
            if predictor is None:predictor=Predictor()
            mask,report=predictor.predict(image)
    except ValueError as e:raise HTTPException(400,str(e))
    except FileNotFoundError as e:raise HTTPException(503,str(e))
    key=uuid.uuid4().hex;p=REVIEWS/key;p.mkdir(parents=True)
    image.save(p/'input.png');Image.fromarray(mask.astype('uint8')*255).save(p/'prediction.png')
    Image.fromarray(mask.astype('uint8')*255).save(p/'mask.png');overlay(image,mask).save(p/'overlay.png')
    report.update({'review_id':key,'filename':Path(file.filename or 'upload').name,'status':'pending_review',
                   'created_at':datetime.now(timezone.utc).isoformat(),'revision':0,'model_area_pixels':report['area_pixels']})
    save_report(p,report);return public_report(p)

@app.get('/api/reviews/{key}')
def get_review(key:str):return public_report(folder(key))

@app.post('/api/reviews/{key}/accept')
def accept(key:str):
    with LOCK:
        p=folder(key);report=json.loads((p/'report.json').read_text())
        report['status']='accepted';report['reviewed_at']=datetime.now(timezone.utc).isoformat()
        save_report(p,report);return public_report(p)

@app.post('/api/reviews/{key}/correct')
async def correct(key:str,mask:UploadFile=File(...)):
    image=read_image(await mask.read(12*1024*1024+1))
    with LOCK:
        p=folder(key);report=json.loads((p/'report.json').read_text());source=Image.open(p/'input.png').convert('L')
        if image.size!=source.size:raise HTTPException(400,'Correction must match the original image dimensions.')
        edited=np.array(image)>127;x0,y0,x1,y1=report['crop_box_xyxy']
        # Explicitly mark excluded border pixels as unmeasured, never editable defect area.
        edited[:y0]=False;edited[y1:]=False;edited[:,:x0]=False;edited[:,x1:]=False
        original=np.array(Image.open(p/'prediction.png'))>127
        report.update(measures(edited));report['coverage_fraction']=float(edited.sum()/report['evaluated_pixels'])
        report['status']='corrected';report['revision']+=1;report['reviewed_at']=datetime.now(timezone.utc).isoformat()
        report['changed_pixels_from_model']=int(np.count_nonzero(edited!=original))
        Image.fromarray(edited.astype('uint8')*255).save(p/f'correction-{report["revision"]}.png')
        Image.fromarray(edited.astype('uint8')*255).save(p/'mask.png');overlay(source,edited).save(p/'overlay.png')
        save_report(p,report);return public_report(p)

@app.get('/api/reviews/{key}/files/{name}')
def artifact(key:str,name:str):
    if name not in ['input.png','prediction.png','mask.png','overlay.png','report.json','regions.csv','inspection.csv']:raise HTTPException(404)
    p=folder(key)/name
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p,filename=name)

@app.get('/api/reviews/{key}/export')
def export(key:str):
    p=folder(key);buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for file in p.iterdir():
            if file.is_file():z.write(file,file.name)
        z.writestr('README.txt','SEM Review inspection export\nmask.png is the current human-reviewable mask; prediction.png preserves the original model output.\nPixel area is measured only inside crop_box_xyxy; excluded borders are not assessed.\nNo nanometre calibration or production disposition is claimed.\n')
    buf.seek(0)
    return StreamingResponse(buf,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="sem-review-{key[:8]}.zip"'})
