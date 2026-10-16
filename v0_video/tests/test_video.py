import sys,pathlib,tempfile,json,unittest,copy
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from video_adapter import render,predict,uniform_indices
class Tests(unittest.TestCase):
 def test_raw_coordinates_and_no_frames(self):
  row={'framework':'dvd','task':{'query':'q','answer':'SECRET'},'trajectory':[{'kind':'system','content':'rules'},{'kind':'user','content':'q'},{'kind':'assistant','content':'wrong','injected':True},{'kind':'tool','tool_name':'frame_inspect_tool','content':'tool result'}]}
  a,nodes,images,manifest,total=render(row,pathlib.Path('/tmp'),32)
  self.assertEqual([n['step'] for n in nodes],['2','3']);self.assertEqual(nodes[-1]['agent'],'frame_inspect_agent');self.assertNotIn('SECRET',a);self.assertNotIn('injected',a);self.assertEqual(images,[])
  changed=copy.deepcopy(row);changed['ground_truth']={'secret':'gold'};changed['trajectory'][2]['injected']=False;self.assertEqual(a,render(changed,pathlib.Path('/tmp'),32)[0])
 def test_allocation_and_bindings(self):
  with tempfile.TemporaryDirectory() as d:
   base=pathlib.Path(d);(base/'eva').mkdir();Image.new('RGB',(800,600),'red').save(base/'eva/a.png')
   row={'framework':'eva','task':{'query':'q'},'trajectory':[{'kind':'system','content':'r'},{'kind':'assistant','content':'call'},{'kind':'tool','tool_name':'frame_select','content':'frames','frames':[{'index':i,'time_s':i*10,'path':'a.png'} for i in range(50)]}]}
   state,nodes,imgs,metas,total=render(row,base,4);self.assertEqual(total,50);self.assertEqual(len(imgs),4);self.assertEqual(metas[0]['time_s'],0);self.assertEqual(metas[-1]['time_s'],490);self.assertTrue(all(m['message_index']==2 for m in metas));self.assertEqual(metas[0]['sent_size'],(448,336));self.assertNotIn('data:image',state)
 def test_missing_asset_explicit(self):
  row={'framework':'eva','task':{},'trajectory':[{'kind':'tool','tool_name':'frame_select','frames':[{'path':'missing.png'}]}]}
  with self.assertRaises(FileNotFoundError):render(row,pathlib.Path('/tmp'),32)
 def test_frame_policy(self):
  self.assertEqual(uniform_indices(3,32),[0,1,2]);self.assertEqual(uniform_indices(950,32)[-1],949);self.assertEqual(len(set(uniform_indices(950,32))),32)
if __name__=='__main__':unittest.main()
