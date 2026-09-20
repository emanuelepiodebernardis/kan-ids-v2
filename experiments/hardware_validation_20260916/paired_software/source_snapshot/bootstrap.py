"""Create an isolated Python environment, test the kit, then run the fixed five-seed study."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent


def execute(argv, log, env):
    line = '\n> ' + subprocess.list2cmdline([str(a) for a in argv]) + '\n'
    print(line, flush=True)
    log.write(line)
    log.flush()
    with subprocess.Popen([str(a) for a in argv], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                          errors='replace', bufsize=1) as p:
        try:
            for line in p.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = p.wait()
        except BaseException:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            raise
    if code:
        raise RuntimeError('Command failed, exit code ' + str(code))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_dir = ROOT / 'runs' / ('SETUP_' + stamp + '_' + uuid.uuid4().hex[:8])
    log_dir.mkdir(parents=True, exist_ok=False)
    try:
        if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
            raise RuntimeError('Run with 64-bit Python 3.11: py -3.11 bootstrap.py')
        from run_stage2 import verify_kit
        verify_kit()
        env = dict(os.environ)
        env.update({'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
        local_python = ROOT / 'runtime' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        prior_python = ROOT.parent / 'KAN_IDS_STAGE1_20260916_v0.13.0' / 'runtime' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        python = prior_python if prior_python.is_file() else local_python
        with (log_dir / 'setup.log').open('w', encoding='utf-8', newline='\n') as log:
            log.write('Bootstrap Python: ' + sys.version + '\nPlatform: ' + platform.platform() + '\n')
            if not python.exists():
                execute([sys.executable, '-m', 'venv', ROOT / 'runtime'], log, env)
            if python == local_python:
                execute([python, '-m', 'pip', 'install', '--disable-pip-version-check',
                         '--only-binary=:all:', '-r', ROOT / 'requirements.txt'], log, env)
            else:
                log.write('Reusing the isolated Stage1 runtime read-only; exact package versions are checked by Stage2.\n')
            execute([python, '-m', 'unittest', 'discover', '-s', ROOT / 'tests', '-v'], log, env)
            log.flush()
            command = [python, '-u', ROOT / 'run_stage2.py', '--setup-log', log_dir / 'setup.log']
            project = ROOT.parent / 'work'
            if (project / 'kanids' / 'models.py').is_file():
                command += ['--project', project]
            execute(command, log, env)
        print('\nDONE. Send the PAIR_STAGE2_*.zip printed above.', flush=True)
        return 0
    except BaseException as exc:
        detail = traceback.format_exc()
        (log_dir / 'error.txt').write_text(detail, encoding='utf-8', newline='\n')
        target = log_dir.with_suffix('.zip')
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in log_dir.iterdir():
                if path.is_file():
                    z.write(path, log_dir.name + '/' + path.name)
        print('\nSTOP: ' + str(exc), flush=True)
        print('SEND_SETUP_ZIP: ' + str(target), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
