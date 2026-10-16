import requests,pathlib,json,concurrent.futures,time,hashlib,os
base=pathlib.Path(__file__).resolve().parent;root=base/'model';m=json.loads((base/'manifest.json').read_text());rev=m['revision'];files=m['files'];jobs=[];block=32*1024*1024
for f in files:
 p=root/f['rfilename'];p.parent.mkdir(parents=True,exist_ok=True)
 if f['size']>64*1024*1024:
  if not p.exists():
   with p.open('wb') as out:out.truncate(f['size'])
  with p.open('rb') as inp:
   for start in range(0,f['size'],block):
    end=min(f['size']-1,start+block-1);inp.seek(start);first=inp.read(4096);inp.seek(max(start,end-4095));last=inp.read(4096)
    if not any(first) or not any(last):jobs.append((f,start,end))
 elif not p.exists() or p.stat().st_size!=f['size']:jobs.append((f,None,None))
def get(job):
 f,start,end=job;p=root/f['rfilename']
 for attempt in range(10):
  try:
   r=requests.get(f"https://huggingface.co/vllm-sr/d3/resolve/{rev}/{f['rfilename']}",params={'chunk':start} if start is not None else {},headers={'Range':f'bytes={start}-{end}'} if start is not None else {},timeout=120);r.raise_for_status()
   expected=end-start+1 if start is not None else f['size']
   if len(r.content)!=expected:raise ValueError('wrong size')
   if start is None:p.write_bytes(r.content)
   else:
    with p.open('r+b') as out:out.seek(start);out.write(r.content)
  except Exception:
   if attempt==9:raise
   time.sleep(2+attempt);continue
  return
print('remaining chunks',len(jobs),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
 for i,future in enumerate(concurrent.futures.as_completed([ex.submit(get,j) for j in jobs])):
  future.result()
  if i%50==0:print('completed',i+1,'/',len(jobs),flush=True)
print('Checking SHA256',flush=True)
for f in files:
 p=root/f['rfilename'];assert p.stat().st_size==f['size']
 if 'lfs' in f:
  h=hashlib.sha256()
  with p.open('rb') as inp:
   for chunk in iter(lambda:inp.read(8*1024*1024),b''):h.update(chunk)
  assert h.hexdigest()==f['lfs']['sha256'],f['rfilename']
  print('verified',f['rfilename'],flush=True)
(base/'verified.json').write_text(json.dumps({'revision':rev,'files':len(files),'bytes':sum(f['size'] for f in files),'sha256':'passed'},indent=2));print('DONE verified',flush=True)
