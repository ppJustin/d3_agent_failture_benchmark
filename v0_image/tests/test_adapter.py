import unittest,pathlib,sys,base64,copy
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from image_adapter import render,predict
PIX=base64.b64encode(b'\x89PNG\r\n\x1a\nTEST_IMAGE').decode()
def image():return {'__type__':'PIL.Image','data':PIX}
def example():return {'id':'x','framework':'pixelcraft','task':{'query':'q','answer':'SECRET_ANSWER','images':[image()]},'trajectory':[{'kind':'user','content':'q','images':[image()]},{'kind':'round','round':0,'turns':[{'agent_id':'planner','position':0,'output':'saw figure','images':[image()],'is_injected':True}]},{'kind':'final_answer','content':'result'}],'ground_truth':{'agent':'planner','round':0,'position':0,'mode':'P.1'},'extras':{'answer':'SECRET'}}
class Tests(unittest.TestCase):
 def test_dedup_and_binding(self):
  state,ns,media=render(example());self.assertEqual(len(media.urls),1);self.assertEqual(len(media.attachments),3);self.assertIn('trajectory[1].turns[0].images[0]',str(media.attachments));self.assertIn('IMG_001',state);self.assertNotIn(PIX,state);self.assertNotIn('SECRET',state);self.assertEqual(ns[0]['step'],'0.0')
 def test_no_label_leakage(self):
  a=example();b=copy.deepcopy(a);b['task']['answer']='OTHER';b['ground_truth']={};b['extras']={};b['trajectory'][1]['turns'][0]['is_injected']=False
  s,_,m=render(a);t,_,n=render(b);self.assertEqual(s,t);self.assertEqual(m.urls,n.urls)
 def test_actual_pixels_sent_independent_questions(self):
  def fake(endpoint,payload):
   self.assertEqual(len(payload['images']),1);self.assertTrue(payload['images'][0].startswith('data:image/png;base64,'));self.assertEqual(payload['questions']['step']['criteria'],{'0.0':None})
   return {'answers':{k:{'choice':next(iter(q['criteria'])),'probabilities':{x:float(i==0) for i,x in enumerate(q['criteria'])}} for k,q in payload['questions'].items()}}
  with patch('image_adapter.post',fake):r,_=predict(example());self.assertEqual(r['image_count'],1)
 def test_missing_images_fails(self):
  a=example();a['task'].pop('images');a['trajectory'][0].pop('images');a['trajectory'][1]['turns'][0].pop('images')
  with self.assertRaisesRegex(ValueError,'No embedded'):render(a)
if __name__=='__main__':unittest.main()
