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


def _toolchain():
    """The repository's compiler locator, loaded by path rather than imported.

    `import kanids.toolchain` would go through the package __init__, which
    pulls in numpy, pandas and scikit-learn: dependencies this project does
    not declare and does not need in order to compile C++. The module itself
    only uses os, shutil and pathlib, so it is loaded from its file. When the
    project is extracted outside the repository the file is absent and the
    previous behaviour applies unchanged.

    Why it matters here: g++ is a driver that spawns `as` and `ld`, which it
    looks for on PATH. Invoking an absolute compiler path while its directory
    is NOT on PATH fails with

        g++.exe: fatal error: cannot execute 'as'

    which reads like a broken compiler and is in fact a compiler that cannot
    reach its own assembler. `ambiente()` puts that directory on the PATH of
    the subprocess only.
    """
    import importlib.util
    percorso=Path(__file__).resolve().parents[6]/'kanids'/'toolchain.py'
    if not percorso.is_file():return None
    spec=importlib.util.spec_from_file_location('kanids_toolchain',percorso)
    modulo=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class C3Protocol(unittest.TestCase):
 def test_five_models_all_recovery_paths(self):
  tc=_toolchain()
  compiler=tc.trova('g++') if tc else (os.environ.get('CXX') or shutil.which('g++'))
  if not compiler:self.skipTest(tc.motivo_assenza('g++') if tc else 'Host compiler absent')
  env=tc.ambiente('g++') if tc else None
  with tempfile.TemporaryDirectory(prefix='ram500_protocol_') as tmp:
   for variant,flag in VARIANTS.items():
    exe=Path(tmp)/(variant+('.exe' if os.name=='nt' else ''))
    args=[compiler,'-std=c++11','-Os','-fno-lto','-DHOST_CHECK','-DHW500_C3','-D'+flag,'-I'+str(PROJECT/'include'),'-I'+str(PROJECT/'tests'),str(PROJECT/'tests/c3_protocol_harness.cpp'),'-o',str(exe)]
    built=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',env=env)
    self.assertEqual(built.returncode,0,built.stderr)
    for scenario in SCENARIOS:
     with self.subTest(model=variant,scenario=scenario):
      run=subprocess.run([str(exe),scenario],capture_output=True,text=True,encoding='utf-8',env=env)
      self.assertEqual(run.returncode,0,run.stderr)
      self.assertIn('HOST_C3_PROTOCOL_PASS',run.stdout)
      print(run.stdout.strip())
if __name__=='__main__':unittest.main()
