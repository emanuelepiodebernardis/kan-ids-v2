#!/usr/bin/env python3
"""Private, pinned dependency bootstrap; never changes the global Python install."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import venv

PINS={'platformio':'6.1.19','pyserial':'3.5','numpy':'2.3.5','scipy':'1.17.0'}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',choices=('mega','c3'),required=True)
    args,extra=parser.parse_known_args(argv)
    root=Path(__file__).resolve().parent
    if '--dry-run' in extra:
        from run_continuous import main as run
        return run(['--board',args.board]+extra)
    if sys.version_info[:2]!=(3,11) or sys.maxsize<=2**32:
        print('Нужен Python 3.11 64-bit. Запуск: py -3.11 bootstrap.py --board '+args.board); return 1
    from run_suite import verify_sources,make_archive,write_json,digest
    # Inspect before creating an environment or running package code.
    verify_sources(root)
    bid=datetime.now(timezone.utc).strftime('BOOTSTRAP_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]
    directory=root/'bootstrap_runs'/bid; directory.mkdir(parents=True,exist_ok=False)
    logfile=directory/'bootstrap.log'; receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'incomplete','pins':PINS,'commands':[]}
    env=os.environ.copy(); env['PYTHONUTF8']='1'; env['PYTHONIOENCODING']='utf-8'
    private=root/'.venv'; executable=private/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    code=1
    def run(command,stream):
        print('> '+subprocess.list2cmdline(command),flush=True); stream.write('> '+subprocess.list2cmdline(command)+'\n');stream.flush()
        proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
        try:
            for line in proc.stdout: stream.write(line);stream.flush();print(line,end='',flush=True)
            returncode=proc.wait()
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
        finally:proc.stdout.close()
        receipt['commands'].append({'argv':command,'returncode':returncode})
        if returncode: raise RuntimeError('Команда завершилась с ошибкой '+str(returncode))
    try:
        with logfile.open('x',encoding='utf-8',newline='\n') as stream:
            if not executable.is_file():
                print('Создаю отдельное окружение пакета; глобальный Python не меняется.',flush=True)
                stream.write('Create private venv with system_site_packages=True\n');stream.flush()
                venv.EnvBuilder(with_pip=True,system_site_packages=True).create(private)
            probe="import importlib.metadata as m,json; names=['platformio','pyserial','numpy','scipy']; result={};\nfor n in names:\n try: result[n]=m.version(n)\n except m.PackageNotFoundError: result[n]=None\nprint(json.dumps(result))"
            result=subprocess.run([str(executable),'-c',probe],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,check=True)
            observed=json.loads(result.stdout);receipt['initial_versions']=observed
            missing=[name+'=='+version for name,version in PINS.items() if observed.get(name)!=version]
            if missing: run([str(executable),'-m','pip','install','--disable-pip-version-check']+missing,stream)
            else: print('Все закреплённые зависимости уже доступны; скачивание не требуется.',flush=True)
            run([str(executable),'-c',"from run_suite import check_versions; print(check_versions())"],stream)
            run([str(executable),'-m','unittest','discover','-s','tests_runner','-v'],stream)
            run([str(executable),'-m','unittest','discover','-s','tests_measurement','-v'],stream)
            run([str(executable),'-m','unittest','discover','-s','tests_summary','-v'],stream)
            run([str(executable),'-m','unittest','discover','-s','tests_continuous','-v'],stream)
            receipt['status']='bootstrap_and_tests_pass'
        write_json(directory/'BOOTSTRAP_RECORD.json',receipt);make_archive(directory)
        command=[str(executable),str(root/'run_continuous.py'),'--board',args.board,'--bootstrap-evidence',str(directory)]+extra
        # The runner captures hardware commands itself; the setup receipt is copied into its session.
        code=subprocess.call(command,cwd=root,env=env)
        receipt['runner_returncode']=code
    except (Exception,KeyboardInterrupt) as exc:
        receipt['error']=type(exc).__name__+': '+str(exc);print('\nОСТАНОВКА: '+receipt['error'],file=sys.stderr)
    finally:
        receipt['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(directory/'BOOTSTRAP_RECORD.json',receipt)
        archive=make_archive(directory)
        if code:print('Логи подготовки: '+str(archive))
    return code

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(errors='replace');sys.stderr.reconfigure(errors='replace')
    raise SystemExit(main())
