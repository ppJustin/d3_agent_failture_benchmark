#!/usr/bin/env python3
"""Manage local D3 replicas, sharing model files and the d3_v1 environment."""
import argparse,json,os,pathlib,signal,socket,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent

def health(host,port):
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/health',timeout=3) as r:return json.load(r)
    except Exception:return None

def owners(port):
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/'cmdline').read_bytes().split(b'\0');args=[v.decode() for v in cmd if v]
            if str(ROOT/'model/d3_server.py') not in args:continue
            if '--port'not in args or args[max(i for i,v in enumerate(args) if v=='--port')+1]!=str(port):continue
            env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'='in v)
            found.append({'pid':int(p.name),'gpu':env.get(b'CUDA_VISIBLE_DEVICES',b'?').decode(),'port':port})
        except (OSError,ValueError):continue
    return found

def main():
    ap=argparse.ArgumentParser(description='D3: explicitly select physical GPU; reuse local d3_v1 + weights.')
    ap.add_argument('action',choices=['start','status','stop'],nargs='?',default='start');ap.add_argument('--gpu',type=int,required=True,help='Physical GPU index shown by nvidia-smi');ap.add_argument('--port',type=int,help='Default: 8899 + GPU index');ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--python',default='/cpfs01/zijian.pan/conda-envs/d3_v1/bin/python');ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--wait',type=int,default=120,help='Seconds to wait for health; timeout leaves process running');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if a.gpu<0:ap.error('--gpu must be nonnegative')
    port=a.port if a.port is not None else 8899+a.gpu
    if not 1<=port<=65535 or a.batch_size<1 or a.wait<0:ap.error('Invalid port, batch-size or wait')
    host='127.0.0.1' if a.host=='0.0.0.0' else a.host
    processes=owners(port)
    if a.action=='status':
        print(json.dumps({'requested_gpu':a.gpu,'processes':processes,'health':health(host,port)},indent=2));return
    if a.action=='stop':
        for p in processes:
            if p['gpu']!=str(a.gpu):raise SystemExit(f"Port belongs to GPU {p['gpu']}; refusing to stop another GPU")
        if not processes:print('No matching D3 process');return
        for p in processes:os.kill(p['pid'],signal.SIGTERM);print('Sent SIGTERM to',p['pid'])
        return
    command=[a.python,'-u',str(ROOT/'model/d3_server.py'),'--model',str(ROOT/'model'),'--device','cuda:0','--batch-size',str(a.batch_size),'--host',a.host,'--port',str(port)]
    logfile=ROOT/'logs'/f'gpu{a.gpu}_port{port}.log'
    if a.dry_run:
        print(json.dumps({'CUDA_VISIBLE_DEVICES':str(a.gpu),'command':command,'log':str(logfile)},indent=2));return
    if processes:
        if any(p['gpu']!=str(a.gpu) for p in processes):raise SystemExit('Port already belongs to a different GPU')
        print('Already running:',json.dumps(processes));print(json.dumps(health(host,port)));return
    with socket.socket() as probe:
        try:probe.bind((a.host,port))
        except OSError as e:raise SystemExit(f'Port {port} unavailable: {e}')
    if not pathlib.Path(a.python).is_file() or not (ROOT/'model/d3_server.py').is_file():raise SystemExit('Missing Python environment or model server')
    listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus={int(line.split(',')[0]):int(line.split(',')[1]) for line in listing.strip().splitlines()}
    if a.gpu not in gpus:raise SystemExit(f'GPU {a.gpu} not found; available {sorted(gpus)}')
    if gpus[a.gpu]<55000:raise SystemExit(f'GPU {a.gpu} has only {gpus[a.gpu]} MiB free; use an available 80GB GPU')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
    logfile.parent.mkdir(exist_ok=True)
    with logfile.open('ab',buffering=0) as log:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,start_new_session=True,cwd=ROOT)
    (ROOT/'logs'/f'gpu{a.gpu}_port{port}.pid').write_text(str(p.pid)+'\n')
    print(f'Started PID={p.pid}, physical GPU={a.gpu}, port={port}, log={logfile}',flush=True)
    end=time.monotonic()+a.wait
    while time.monotonic()<end:
        if p.poll() is not None:raise SystemExit(f'Server exited {p.returncode}; see {logfile}')
        status=health(host,port)
        if status and status.get('status')=='ready':print(f'READY http://{host}:{port}/v1/systemone');return
        time.sleep(2)
    print(f'Still starting; process remains running. Check {logfile} or run status.')
if __name__=='__main__':main()
