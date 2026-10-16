import unittest
from rescore import convert,score,cell_metrics
class Tests(unittest.TestCase):
 def test_coordinate_translation(self):
  def p(s,a='agent'):return {'step':s,'agent_choice':a,'mode':'R.2'}
  self.assertEqual(convert(p('3.1','assistant'),'mathchat')['step_coord'],'7')
  self.assertEqual(convert(p('6','orchestrator'),'dvd')['step_coord'],'4')
  self.assertEqual(convert(p('6'),'eva')['step_coord'],'2')
  self.assertEqual(convert(p('7'),'eva')['step_coord'],'2')
  self.assertEqual(convert(p('5'),'agentoccam')['agent_name'],'web_agent')
 def test_accepted_alternates_independent(self):
  gt={'agent':'node_1','round':1,'position':0,'mode':'C.3','accepted_predictions':[{'agent_name':'node_1','step_coord':'1.1','mode':'R.1'}]}
  self.assertTrue(all(score({'agent_name':'node_1','step_coord':'1.1','error_mode':'C.3'},gt,'macnet').values()))
 def test_official_macro_and_single_agent(self):
  def r(fw,a,s,m,g,p):return dict(fw=fw,a_correct=a,s_correct=s,m_correct=m,gt_mode=g,pr_mode=p)
  rows=[r('debate',True,True,True,'V.1','V.1'),r('smolagents',True,False,False,'R.1','V.1'),r('smolagents',True,False,False,'R.1','V.1')]
  m=cell_metrics(rows);self.assertEqual(m['Who'],1);self.assertEqual(m['When'],.5);self.assertEqual(m['All'],.5);self.assertEqual(m['What'],.25)
if __name__=='__main__':unittest.main()
