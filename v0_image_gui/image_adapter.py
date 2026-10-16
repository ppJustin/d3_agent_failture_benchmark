"""Whole image trajectory -> sanitized text + ordered actual images, no gold leakage."""
import base64,hashlib,json,pathlib,time,math,io
from PIL import Image
from attribute import decode,prepare,post,match_label,ALLOWED
from text_reference import render as render_text

class Images:
    def __init__(self):self.urls=[];self.metadata=[];self.by_hash={};self.attachments=[]
    def replace(self,value,source):
        if isinstance(value,dict):
            if value.get('__type__')=='PIL.Image':
                encoded=value.get('data')
                if not isinstance(encoded,str):raise ValueError(f'Invalid embedded image at {source}')
                if encoded.startswith('data:'):encoded=encoded.split(',',1)[1]
                raw=base64.b64decode(encoded,validate=True)
                original_sha=hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as probe:
                    original_size=probe.size
                    resized=False
                    if probe.width*probe.height>16_000_000:
                        scale=math.sqrt(15_900_000/(probe.width*probe.height))
                        size=(max(1,int(probe.width*scale)),max(1,int(probe.height*scale)))
                        converted=probe.convert('RGB').resize(size,Image.Resampling.LANCZOS)
                        buf=io.BytesIO();converted.save(buf,format='PNG');raw=buf.getvalue();encoded=base64.b64encode(raw).decode();resized=True
                    else:size=original_size
                sha=hashlib.sha256(raw).hexdigest()
                if raw.startswith(b'\x89PNG\r\n\x1a\n'):mime='image/png'
                elif raw.startswith(b'\xff\xd8'):mime='image/jpeg'
                elif raw[:4]==b'RIFF' and raw[8:12]==b'WEBP':mime='image/webp'
                else:raise ValueError(f'Unsupported image bytes at {source}')
                if sha not in self.by_hash:
                    index=len(self.urls)+1;self.by_hash[sha]=index
                    self.urls.append(f'data:{mime};base64,'+encoded)
                    self.metadata.append({'image_id':f'IMG_{index:03d}','images_index':index-1,'sha256':sha,'bytes':len(raw),'mime':mime,'original_sha256':original_sha,'original_size':original_size,'sent_size':size,'resized_for_server_limit':resized})
                index=self.by_hash[sha];ref=f'IMG_{index:03d}'
                self.attachments.append({'image_id':ref,'source':source})
                return f'[Image {ref}; attached image #{index}]'
            return {k:self.replace(v,source+'.'+k) for k,v in value.items()}
        if isinstance(value,list):return [self.replace(v,f'{source}[{i}]') for i,v in enumerate(value)]
        if isinstance(value,str) and value.startswith('data:image/'):
            return self.replace({'__type__':'PIL.Image','data':value},source)
        return value

def render(row):
    media=Images();task=decode(row['task']);trajectory=decode(row['trajectory'])
    safe_task={'query':task.get('query','')}
    if task.get('images'):
        refs=media.replace(task['images'],'task.images');safe_task['query']+='\nTask images: '+json.dumps(refs)
    safe=[]
    def event(e,path):
        # Explicit allowlist. Never traverse ground truth, extras or injection metadata.
        out={k:v for k,v in e.items() if k in ALLOWED|{'kind','agent','agent_id','role','round','position','stage','step_number'}}
        out={k:media.replace(v,path+'.'+k) for k,v in out.items()}
        for key in ['images','observation_images']:
            if e.get(key):
                refs=media.replace(e[key],path+'.'+key)
                dest='output' if 'output'in out else 'plan' if 'plan'in out else 'content' if out.get('kind')=='user' else 'observation'
                original=out.get(dest,'')
                out[dest]=(original if isinstance(original,str) else json.dumps(original,ensure_ascii=False))+'\n'+key+': '+json.dumps(refs)
        if 'turns'in e:out['turns']=[event(t,f'{path}.turns[{j}]') for j,t in enumerate(e['turns'])]
        return out
    for i,e in enumerate(trajectory):safe.append(event(e,f'trajectory[{i}]'))
    state,nodes=render_text({'framework':row['framework'],'task':safe_task,'trajectory':safe})
    if not media.urls:raise ValueError('No embedded supported images found; refusing silent text-only evaluation')
    legend='\n'.join(f"Image #{i+1} = {m['image_id']}" for i,m in enumerate(media.metadata))
    state=('The attached images appear before this text in the order listed below. Image references in the '
           'transcript indicate which image was present at the task or execution step. Repeated references '
           'refer to the same image. Later-step images are retrospective evidence; do not assume an agent '
           'saw an image before its recorded occurrence.\n'+legend+'\n\n'+state)
    return state,nodes,media

def predict(row,endpoint='http://127.0.0.1:8901/v1/systemone'):
    state,nodes,media=render(row);taxonomy=json.loads(pathlib.Path(__file__).with_name('taxonomy.json').read_text())
    qs={'agent':{'type':'choice','instructions':"Which agent's turn first introduced the decisive error in this failed run?",'criteria':dict.fromkeys(n['agent'] or 'agent' for n in nodes)},'step':{'type':'choice','instructions':'At which step coordinate did the first decisive error occur?','criteria':dict.fromkeys(n['step'] for n in nodes)},'mode':{'type':'choice','instructions':'Which taxonomy error mode best matches the first decisive error?','criteria':{k:v['name']+': '+v['description'] for k,v in taxonomy.items()}}}
    start=time.monotonic();raw=post(endpoint,{'model':'d3','state':state,'images':media.urls,'questions':qs})
    answers=raw['answers']
    for axis,q in qs.items():
        a=answers[axis];p=a['probabilities']
        if a['choice']not in q['criteria'] or set(p)!=set(q['criteria']):raise ValueError('Invalid choices')
        if any(not math.isfinite(v) or v<0 for v in p.values()) or abs(sum(p.values())-1)>.01:raise ValueError('Invalid probabilities')
    agent=answers['agent']['choice'];step=answers['step']['choice'];mode=answers['mode']['choice'];single=row['framework'] in {'agentoccam','gemini','openai_cua'}
    result={'prediction':{'agent':None if single else agent,'agent_choice':agent,'step':step,'mode':mode},'raw_response':raw,'probabilities':{k:a['probabilities'] for k,a in answers.items()},'image_count':len(media.urls),'image_occurrences':len(media.attachments),'images':media.metadata,'image_attachments':media.attachments,'agent_step_consistent':any(n['step']==step and (n['agent'] or 'agent')==agent for n in nodes),'latency_s':time.monotonic()-start,'state_sha256':hashlib.sha256(state.encode()).hexdigest(),'truncated':False,'criteria':{k:q['criteria'] for k,q in qs.items()}}
    if 'ground_truth'in row:
        g=decode(row['ground_truth']);a=g.get('agent') is None and 'agents'not in g or agent in g.get('agents',[g.get('agent')]);s=any(match_label(n,g,row['framework'])[1] for n in nodes if n['step']==step);m=mode==g['mode']
        result.update(ground_truth=g,score={'agent':bool(a),'step':s,'mode':m,'all_three':bool(a and s and m)},single_agent=single)
    return result,{'model':'d3','state':state,'questions':qs,'images':media.metadata,'attachments':media.attachments}
