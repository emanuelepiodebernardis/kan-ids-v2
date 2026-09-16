"""Host-only state/transport regression: no inference that physical RAM is valid."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = {'coeff':'HB_COEFF','lut':'HB_LUT14','mlp':'HB_MLP','kanml':'HB_MLCOEFF','dt5':'HB_DT5'}
SCENARIOS = ('replay','invalid','disconnect','partial','short','control_partial','task_create_fail','task_timeout','heap_control_fail','stack_control_fail')
class C3Protocol(unittest.TestCase):
 def test_five_models_all_recovery_paths(self):
  compiler=os.environ.get('CXX') or shutil.which('g++')
  if not compiler:self.skipTest('Host compiler absent')
  with tempfile.TemporaryDirectory(prefix='ram500_protocol_') as tmp:
   for variant,flag in VARIANTS.items():
    exe=Path(tmp)/(variant+('.exe' if os.name=='nt' else ''))
    args=[compiler,'-std=c++11','-Os','-fno-lto','-DHOST_CHECK','-DHW500_C3','-D'+flag,'-I'+str(PROJECT/'include'),'-I'+str(PROJECT/'tests'),str(PROJECT/'tests/c3_protocol_harness.cpp'),'-o',str(exe)]
    built=subprocess.run(args,capture_output=True,text=True,encoding='utf-8')
    self.assertEqual(built.returncode,0,built.stderr)
    for scenario in SCENARIOS:
     with self.subTest(model=variant,scenario=scenario):
      run=subprocess.run([str(exe),scenario],capture_output=True,text=True,encoding='utf-8')
      self.assertEqual(run.returncode,0,run.stderr)
      self.assertIn('HOST_C3_PROTOCOL_PASS',run.stdout)
      print(run.stdout.strip())
if __name__=='__main__':unittest.main()
