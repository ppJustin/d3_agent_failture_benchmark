"""Rescore existing D3 outputs with the unmodified official scorer/leaderboard."""
import collections,csv,hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/'references/whowhen_pro_official'
sys.path.insert(0,str(OFFICIAL))
from whowhen_eval.score import score
from whowhen_eval.leaderboard import cell_metrics,composite
SOURCES={'text':'v0_test/results/predictions.jsonl','image':'v0_image/results/full/predictions.jsonl','image_gui':'v0_image_gui/results/full/predictions.jsonl','video':'v0_video/results/full/predictions.jsonl'}

def convert(pred,fw):
    # Pure coordinate/label translations based on our input convention, never gold.
    agent=pred.get('agent_choice',pred.get('agent'));step=str(pred['step'])
    if fw=='mathchat':
        r,p=map(int,step.split('.'));step=str(2*r+p)
    elif fw=='dvd':step=str(int(step)-2)
    elif fw=='eva':step=str((int(step)-2)//2)
    if fw in {'agentoccam','gemini'} and agent=='agent':agent='web_agent'
    elif fw=='openai_cua' and agent=='agent':agent='computer_use_agent'
    return {'agent_name':agent,'step_coord':step,'error_mode':pred['mode']}

def cellrow(r):
    sc=r['score'];return {'fw':r['framework'],'modality':r['modality'],'a_correct':sc['agent'],'s_correct':sc['step'],'m_correct':sc['mode'],'gt_mode':r['ground_truth']['mode'],'pr_mode':(r.get('prediction') or {}).get('error_mode','')}

def main():
    out=ROOT/'official_eval';dest=out/'results/d3';dest.mkdir(parents=True,exist_ok=True)
    report={'official_repository':'https://github.com/ag2ai/whowhen_pro','official_commit':subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip(),'sources':{},'per_modality':{}}
    for modality,source in SOURCES.items():
        path=ROOT/source;latest={}
        with path.open() as f:
            for line in f:
                r=json.loads(line);latest[r['trace_id']]=r
        records=[];changes=collections.Counter()
        with (dest/(modality+'.jsonl')).open('w') as f:
            for r in latest.values():
                failed=bool(r.get('error'));pred=None if failed else convert(r['prediction'],r['framework'])
                sc=score(pred,r['ground_truth'],r['framework'])
                item={'trace_id':r['trace_id'],'framework':r['framework'],'modality':modality,'ground_truth':r['ground_truth'],'prediction':pred,'score':sc}
                if failed:item['error']=r['error']
                else:
                    for axis in ['agent','step','mode']:
                        if sc[axis]!=r['score'][axis]:changes[axis]+=1
                    if all(sc.values())!=r['score']['all_three']:changes['all_three']+=1
                records.append(item);f.write(json.dumps(item,ensure_ascii=False)+'\n')
        rows=[cellrow(r) for r in records];valid=[cellrow(r) for r in records if not r.get('error')]
        groups=collections.defaultdict(list)
        for r in valid:groups[r['fw']].append(r)
        report['per_modality'][modality]={'total':len(rows),'successful':len(valid),'errors':len(rows)-len(valid),'coverage':len(valid)/len(rows),'official_success_only':cell_metrics(valid),'failure_inclusive_supplement':cell_metrics(rows),'changed_verdicts_vs_original':dict(changes),'per_framework_success_only':{fw:{'n':len(v),'metrics':cell_metrics(v)} for fw,v in groups.items()}}
        report['sources'][modality]={'path':source,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report['composite_official_success_only']=composite({k:v['official_success_only'] for k,v in report['per_modality'].items()})
    report['composite_failure_inclusive_supplement']=composite({k:v['failure_inclusive_supplement'] for k,v in report['per_modality'].items()})
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (out/'summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['modality','total','successful','errors','Who_percent','When_percent','What_macro_F1_percent','All_percent'])
        for k,v in report['per_modality'].items():w.writerow([k,v['total'],v['successful'],v['errors']]+[None if v['official_success_only'][a] is None else 100*v['official_success_only'][a] for a in ['Who','When','What','All']])
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
