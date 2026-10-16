import sys,pathlib,json,unittest,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from attribute import prepare,match_label
DATA=pathlib.Path(__file__).resolve().parents[2]/'whowhen_pro/data/text.jsonl'
class Tests(unittest.TestCase):
 def test_all_nodes_cover_targets_and_remove_leakage(self):
  for line in DATA.read_text().splitlines():
   r=json.loads(line);state,nodes=prepare(r);g=json.loads(r['ground_truth'])
   if not any(all(match_label(n,g,r['framework'])) for n in nodes):
    self.assertEqual(r['framework'],'smolagents',r['id'])
   modified=copy.deepcopy(r);modified['ground_truth']='{"secret":"LABEL_SECRET"}';task=json.loads(r['task']);task['answer']='ANSWER_SECRET';modified['task']=json.dumps(task);modified['extras']='{"secret":"EXTRAS_SECRET"}';modified['id']='SECRET_ID'
   t=json.loads(r['trajectory'])
   def alter(v):
    if isinstance(v,dict):
     for k in list(v):
      if k in {'is_injected','is_replayed','is_live','phase','injection_via'}:v[k]='SECRET_INJECTION'
      else:alter(v[k])
    elif isinstance(v,list):
     for x in v:alter(x)
   alter(t);modified['trajectory']=json.dumps(t)
   self.assertEqual(state,prepare(modified)[0],r['id'])
 def test_nested_turn_sources(self):
  r={'framework':'debate','task':{'query':'q'},'trajectory':[{'kind':'user','content':'q'},{'kind':'round','round':2,'turns':[{'agent_id':'a','output':'x'},{'agent_id':'b','output':'y'}]}]}
  _,nodes=prepare(r);self.assertEqual(nodes[1]['step'],'2.1');self.assertEqual(nodes[1]['trajectory_index'],1);self.assertEqual(nodes[1]['turn_index'],1)
if __name__=='__main__':unittest.main()
