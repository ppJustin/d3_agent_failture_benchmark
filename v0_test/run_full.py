"""Resumable full-text independent D3 evaluation; never copies input traces."""
import argparse,collections,csv,fcntl,hashlib,json,os,pathlib,time,traceback
from run_jev_style import predict
ROOT=pathlib.Path(__file__).resolve().parent

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def summarize(records,total,started):
    rs=list(records.values());groups=collections.defaultdict(list)
    for r in rs:groups[r['framework']].append(r)
    by={}
    for fw,vs in groups.items():
        by[fw]={'n':len(vs),'success':sum('error'not in r for r in vs)}
        for axis in ['agent','step','mode','all_three']:by[fw][axis]=sum(r.get('score',{}).get(axis,False) for r in vs)/len(vs)
    multi=[r for r in rs if r['framework']not in {'alfagent','smolagents'}]
    result={'total_expected':total,'attempted':len(rs),'successful':sum('error'not in r for r in rs),'errors':sum('error'in r for r in rs),'complete':len(rs)==total,'scoring':'Custom coordinate scorer. Official-style aggregation is NOT verified official scoring. Failures count as wrong.','per_framework':by,'agent_multi_micro':sum(r.get('score',{}).get('agent',False) for r in multi)/len(multi) if multi else None,'inconsistent_pairs':sum(not r.get('agent_step_consistent',True) for r in rs),'elapsed_this_run_s':time.time()-started,'mean_success_latency_s':sum(r.get('latency_s',0) for r in rs)/max(1,sum('error'not in r for r in rs))}
    for axis in ['step','mode','all_three']:result[axis+'_micro']=sum(r.get('score',{}).get(axis,False) for r in rs)/max(1,len(rs))
    labels=json.loads((ROOT/'taxonomy.json').read_text());f1=[]
    for label in labels:
        tp=sum(r.get('prediction',{}).get('mode')==label and r.get('ground_truth',{}).get('mode')==label for r in rs);fp=sum(r.get('prediction',{}).get('mode')==label and r.get('ground_truth',{}).get('mode')!=label for r in rs);fn=sum(r.get('ground_truth',{}).get('mode')==label and r.get('prediction',{}).get('mode')!=label for r in rs);f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    def avg(axis,multi_only=False):
        cells=[v[axis] for fw,v in by.items() if not multi_only or fw not in {'alfagent','smolagents'}]
        return sum(cells)/len(cells) if cells else None
    result['framework_macro_custom']={'Who':avg('agent',True),'When':avg('step'),'What_macro_f1_17':sum(f1)/len(f1),'All':avg('all_three')}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=pathlib.Path,default=ROOT.parent/'whowhen_pro/data/text.jsonl');ap.add_argument('--output-dir',type=pathlib.Path,default=ROOT/'results');ap.add_argument('--endpoint',default='http://127.0.0.1:8899/v1/systemone');ap.add_argument('--retry-errors',action='store_true');ap.add_argument('--limit',type=int);args=ap.parse_args()
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True);lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();data=args.input.read_bytes();rows=[json.loads(l) for l in data.splitlines() if l.strip()]
    if args.limit:rows=rows[:args.limit]
    manifest={'input_path':str(args.input.resolve()),'input_sha256':hashlib.sha256(data).hexdigest(),'n':len(rows),'endpoint':args.endpoint,'strategy':'independent_agent_step_mode','code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'attribute.py',ROOT/'run_jev_style.py',ROOT/'run_full.py',ROOT/'taxonomy.json']},'framework_counts':dict(collections.Counter(r['framework'] for r in rows))}
    mp=root/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Manifest differs; choose a new output directory')
    atomic(mp,manifest);records={};out=root/'predictions.jsonl'
    if out.exists():
        with out.open('rb+') as inp:
            while True:
                offset=inp.tell();line=inp.readline()
                if not line:break
                try:r=json.loads(line);records[r['trace_id']]=r
                except json.JSONDecodeError:
                    if inp.read():raise ValueError('Corrupt non-final record')
                    inp.truncate(offset);break
    print('START',json.dumps(manifest),flush=True)
    with out.open('a') as fh:
        for row in rows:
            old=records.get(row['id'])
            if old and (not args.retry_errors or 'error'not in old):continue
            record={'trace_id':row['id'],'framework':row['framework'],'benchmark':row['benchmark'],'ground_truth':json.loads(row['ground_truth'])}
            try:result,_,_=predict(row,args.endpoint);record.update(result)
            except Exception as exc:record['error']=f'{type(exc).__name__}: {exc}';record['traceback']=traceback.format_exc()
            fh.write(json.dumps(record,ensure_ascii=False)+'\n');fh.flush();os.fsync(fh.fileno());records[row['id']]=record
            print(f"{len(records)}/{len(rows)}",row['id'],record.get('score',record.get('error')),flush=True)
            # Frequent cheap progress updates; full metric pass every 100 samples.
            atomic(root/'progress.json',{'attempted':len(records),'expected':len(rows),'last_id':row['id'],'elapsed_s':time.time()-started,'updated_unix':time.time()})
            if len(records)%100==0:atomic(root/'summary.json',summarize(records,len(rows),started))
    summary=summarize(records,len(rows),started);atomic(root/'summary.json',summary)
    with (root/'comparison.csv').open('w',newline='') as fh:
        w=csv.writer(fh);w.writerow(['id','framework','agent','step','mode','agent_correct','step_correct','mode_correct','all_three','error'])
        for r in records.values():w.writerow([r['trace_id'],r['framework'],*[r.get('prediction',{}).get(k) for k in ['agent','step','mode']],*[r.get('score',{}).get(k) for k in ['agent','step','mode','all_three']],r.get('error')])
    print('FINISHED',json.dumps(summary),flush=True)
if __name__=='__main__':main()
