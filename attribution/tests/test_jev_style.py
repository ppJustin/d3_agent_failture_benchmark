import sys,pathlib,json,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from unittest.mock import patch
from run_jev_style import predict,render
class IndependentTest(unittest.TestCase):
 def test_independent_agent_is_not_corrected(self):
  row={'framework':'mathchat','task':{'query':'q','answer':'SECRET'},'trajectory':[{'agent_id':'a','round':0,'position':0,'output':'first'},{'agent_id':'b','round':0,'position':1,'output':'second'}],'ground_truth':{'agent':'b','round':0,'position':0,'mode':'R.1'}}
  def fake(endpoint,payload):
   self.assertEqual(set(payload['questions']),{'agent','step','mode'})
   self.assertEqual(payload['questions']['step']['criteria'],{'0.0':None,'0.1':None})
   self.assertNotIn('SECRET',payload['state'])
   return {'answers':{k:{'choice':c,'probabilities':{x:float(x==c) for x in payload['questions'][k]['criteria']}} for k,c in [('agent','b'),('step','0.0'),('mode','R.1')]}}
  with patch('run_jev_style.post',fake):result,_,_=predict(row,'test')
  self.assertEqual(result['prediction']['agent'],'b');self.assertEqual(result['prediction']['step'],'0.0');self.assertFalse(result['agent_step_consistent']);self.assertTrue(result['score']['all_three'])
if __name__=='__main__':unittest.main()
