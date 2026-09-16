"""Real frozen C inference and acquisition state-machine checks under fake clock."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
PROJECT=Path(__file__).resolve().parents[1]
VARIANTS={'coeff':'HB_COEFF','lut':'HB_LUT14','mlp':'HB_MLP','kanml':'HB_MLCOEFF','dt5':'HB_DT5'}
class FirmwareHost(unittest.TestCase):
    def test_all_ten_target_model_paths(self):
        compiler=os.environ.get('CXX') or shutil.which('g++')
        if not compiler:self.skipTest('Host compiler absent; set CXX or install g++')
        with tempfile.TemporaryDirectory(prefix='hw500_') as tmp:
            for board in ['MEGA','C3']:
                for variant,flag in VARIANTS.items():
                    with self.subTest(board=board,variant=variant):
                        exe=Path(tmp)/(board+'_'+variant+('.exe' if os.name=='nt' else ''))
                        args=[compiler,'-std=c++11','-Os','-fno-lto','-DHOST_CHECK','-DHW500_'+board,'-D'+flag,'-I'+str(PROJECT/'include'),'-I'+str(PROJECT/'tests'),str(PROJECT/'tests/host_harness.cpp'),'-o',str(exe)]
                        built=subprocess.run(args,capture_output=True,text=True,encoding='utf-8')
                        self.assertEqual(built.returncode,0,built.stderr)
                        run=subprocess.run([str(exe)],capture_output=True,text=True,encoding='utf-8')
                        self.assertEqual(run.returncode,0,run.stderr)
                        self.assertIn('HOST_FIRMWARE_PASS',run.stdout)
                        print(board,run.stdout.strip())
if __name__=='__main__':unittest.main()
