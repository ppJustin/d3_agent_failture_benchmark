"""Stream existing image JSONL. Default: one example per framework, not a full run."""
import argparse,collections,hashlib,json,pathlib,random,time,fcntl,os
from video_adapter import predict
ROOT=pathlib.Path(__file__).resolve().parent

def select(path,per_framework,seed,all_rows=False):
    if all_rows:
        with path.open() as fh:
            for line in fh:
                if line.strip():yield json.loads(line)
        return
    rng=random.Random(seed);seen=collections.Counter();buckets=collections.defaultdict(list)
    with path.open() as fh:
        for line in fh:
            if not line.strip():continue
            row=json.loads(line);fw=row['framework'];seen[fw]+=1
            if len(buckets[fw])<per_framework:buckets[fw].append(row)
            else:
                j=rng.randrange(seen[fw])
                if j<per_framework:buckets[fw][j]=row
    for fw in sorted(buckets):yield from buckets[fw]

def summary(rs):
    groups=collections.defaultdict(list)
    for r in rs:groups[r['framework']].append(r)
    out={'attempted':len(rs),'success':sum('error'not in r for r in rs),'scoring':'custom coordinates; not verified official scorer','per_framework':{}}
    for fw,vs in groups.items():out['per_framework'][fw]={'n':len(vs),'errors':sum('error'in v for v in vs),**{a:sum(v.get('score',{}).get(a,False) for v in vs)/len(vs) for a in ['agent','step','mode','all_three']}}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets-root',type=pathlib.Path,default=ROOT.parent/'whowhen_pro/video_assets');ap.add_argument('--max-frames',type=int,default=32);ap.add_argument('--input',type=pathlib.Path,default=ROOT.parent/'whowhen_pro/data/video.jsonl');ap.add_argument('--endpoint',default='http://127.0.0.1:8902/v1/systemone');ap.add_argument('--per-framework',type=int,default=1);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--all',action='store_true');ap.add_argument('--id');ap.add_argument('--output-dir',type=pathlib.Path,default=ROOT/'results/smoke');ap.add_argument('--resume',action='store_true');args=ap.parse_args();started=time.time()
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True);lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={'input':str(args.input.resolve()),'assets_root':str(args.assets_root.resolve()),'max_frames':args.max_frames,'endpoint':args.endpoint,'per_framework':args.per_framework,'seed':args.seed,'all':args.all,'id':args.id,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run.py',ROOT/'video_adapter.py',ROOT/'http_utils.py',ROOT/'taxonomy.json']}}
    manifest=root/'manifest.json';out=root/'predictions.jsonl';records={}
    if out.exists() and not args.resume:raise FileExistsError('Use new output-dir or --resume')
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Resume configuration differs')
    manifest.write_text(json.dumps(config,indent=2))
    if out.exists():
        for line in out.open():
            r=json.loads(line);records[r['trace_id']]=r
    progress={'attempted':len(records),'expected':916 if args.all else None,'complete':False}
    iterator=select(args.input,args.per_framework,args.seed,args.all or bool(args.id))
    with out.open('a') as fh:
        for row in iterator:
            if args.id and row['id']!=args.id:continue
            if row['id']in records and 'error'not in records[row['id']]:continue
            record={'trace_id':row['id'],'framework':row['framework'],'benchmark':row['benchmark'],'ground_truth':json.loads(row['ground_truth'])}
            try:
                result,request=predict(row,args.assets_root,args.endpoint,args.max_frames);record.update(result)
                # No image bytes, data URLs, or raw input dataset copies are written.
                (root/(row['id']+'.request_without_pixels.json')).write_text(json.dumps(request,ensure_ascii=False,indent=2))
            except Exception as exc:record['error']=f'{type(exc).__name__}: {exc}'
            records[row['id']]=record;fh.write(json.dumps(record,ensure_ascii=False)+'\n');fh.flush();os.fsync(fh.fileno())
            print(record['trace_id'],record.get('score',record.get('error')),'frames',record.get('sent_frames'),flush=True)
            tmp=root/'summary.tmp';tmp.write_text(json.dumps(summary(list(records.values())),indent=2));tmp.replace(root/'summary.json')
            progress={'attempted':len(records),'expected':916 if args.all else None,'success':sum('error'not in r for r in records.values()),'errors':sum('error'in r for r in records.values()),'last_id':row['id'],'elapsed_s':time.time()-started,'complete':False}
            progress_tmp=root/'progress.tmp';progress_tmp.write_text(json.dumps(progress,indent=2));progress_tmp.replace(root/'progress.json')
            if args.id:break
    if not records:raise ValueError('No matching input samples')
    progress['complete']=True
    (root/'progress.json').write_text(json.dumps(progress,indent=2))
    print(json.dumps(summary(list(records.values())),indent=2),flush=True)
if __name__=='__main__':main()
