"""Download the exact public archive, verify its publisher checksum, extract safely."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request, json, hashlib, zipfile, os, time
ROOT=Path(__file__).resolve().parents[1]
def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    meta=json.load(urllib.request.urlopen('https://zenodo.org/api/records/16895427'))
    (raw/'zenodo_metadata.json').write_text(json.dumps(meta,indent=2))
    entry=next(x for x in meta['files'] if x['key']=='data.zip')
    archive=raw/'carinthia-s.zip'; expected=entry['checksum'].split(':')[-1]
    valid=archive.exists() and hashlib.md5(archive.read_bytes()).hexdigest()==expected
    if not valid:
        total=entry['size']; step=8*1024*1024
        def fetch(start):
            end=min(total-1,start+step-1);path=raw/f'.part-{start}'
            if path.exists() and path.stat().st_size==end-start+1:return path
            req=urllib.request.Request(entry['links']['self'],headers={'Range':f'bytes={start}-{end}'})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req,timeout=180) as response:
                        if response.status!=206:raise RuntimeError('Range request not honored')
                        body=response.read()
                    if len(body)!=end-start+1:raise RuntimeError('Incomplete range')
                    break
                except Exception:
                    if attempt==3:raise
                    time.sleep(2*(attempt+1))
            path.write_bytes(body);print(f'Downloaded {start}-{end}',flush=True);return path
        with ThreadPoolExecutor(max_workers=6) as pool:parts=list(pool.map(fetch,range(0,total,step)))
        tmp=raw/'verified-download.tmp'
        with tmp.open('wb') as out:
            for p in parts:out.write(p.read_bytes())
        if hashlib.md5(tmp.read_bytes()).hexdigest()!=expected:raise RuntimeError('MD5 mismatch')
        os.replace(tmp,archive)
        for p in parts:p.unlink()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            dest=(raw/name).resolve()
            if not dest.is_relative_to(raw.resolve()):raise ValueError('Unsafe archive path')
        z.extractall(raw)
    print('Verified and extracted',archive,flush=True)
if __name__=='__main__':main()
