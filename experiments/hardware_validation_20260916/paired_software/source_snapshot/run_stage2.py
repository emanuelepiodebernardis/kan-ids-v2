"""Create a new, isolated five-seed study run; preserve every previous experiment."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile
from pathlib import Path

from audit_dataset import audit, sha256, write_json

ROOT = Path(__file__).resolve().parent
REQUIRED = {'numpy': '2.3.5', 'pandas': '2.2.3', 'scipy': '1.17.0',
            'scikit-learn': '1.8.0', 'joblib': '1.5.3'}


def verify_kit():
    manifest = json.loads((ROOT / 'KIT_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        p = (ROOT / item['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            raise ValueError('KIT_MISSING_OR_UNSAFE_PATH: ' + item['path'])
        if p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            raise ValueError('KIT_HASH_MISMATCH: ' + item['path'])
    return manifest


def capture_project(path, out):
    report = {'requested_path': str(path) if path else None,
              'used_for_training': False, 'modified': False}
    if path:
        p = Path(path).resolve()
        report['exists'] = p.is_dir()
        selected = ['kanids/models.py', 'kanids/preprocessing.py', 'kanids/config.py',
                    'mcu_pio/include/kan14_coeff_int8.h', 'mcu_pio/platformio.ini']
        report['files'] = {name: sha256(p / name) for name in selected if (p / name).is_file()}
        if (p / '.git').exists() and shutil.which('git'):
            for key, args in [('head', ['rev-parse', 'HEAD']), ('tree', ['rev-parse', 'HEAD^{tree}']),
                              ('status', ['status', '--porcelain', '--untracked-files=no'])]:
                try:
                    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
                    r = subprocess.run(['git', '-C', str(p)] + args, capture_output=True, env=git_env,
                                       text=True, encoding='utf-8', errors='replace', timeout=30)
                    report[key] = {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    report[key] = {'error': str(exc)}
    write_json(out / 'LOCAL_PROJECT_READONLY.json', report)


def archive_run(run):
    files = sorted(p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    write_json(run / 'RUN_MANIFEST.json', {'files': [
        {'path': p.relative_to(run).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in files]})
    target = run.with_suffix('.zip')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                z.write(p, run.name + '/' + p.relative_to(run).as_posix())
    manifest = json.loads((run / 'RUN_MANIFEST.json').read_text(encoding='utf-8'))
    completed = str(json.loads((run / 'RUN_RECORD.json').read_text(encoding='utf-8')).get('status', '')).startswith('STUDY_COMPLETE')
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError('Archived ZIP CRC verification failed')
        for item in manifest['files']:
            content = z.read(run.name + '/' + item['path'])
            if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Archived manifest mismatch: ' + item['path'])
            if completed and item['path'].endswith('.csv.gz'):
                decoded = gzip.decompress(content)
                name = Path(item['path']).name
                expected = {'test_scores.csv.gz': 38088, 'validation_scores.csv.gz': 53623,
                            'split_assignments.csv.gz': 211044}.get(name)
                if expected is not None and len(decoded.splitlines()) != expected:
                    raise ValueError('Archived CSV row count mismatch: ' + item['path'])
    return target


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, help='Optional existing work repo, read-only provenance only')
    ap.add_argument('--audit-only', action='store_true', help='Structural audit only; no scientific dependencies')
    ap.add_argument('--setup-log', type=Path, help='Completed setup/unit-test log to preserve with the run')
    args = ap.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'runs' / ('PAIR_STAGE2_' + stamp + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    record = {'status': 'STARTED', 'created_utc': stamp,
              'hardware_access': False, 'test_predictions_computed': False, 'test_evaluation_planned': True,
              'source_project_modified': False, 'audit_only': args.audit_only}
    exit_code = 1
    try:
        verify_kit()
        print('KIT_MANIFEST_PASS', flush=True)
        if sys.version_info < (3, 11):
            raise RuntimeError('Python 3.11 or newer is required.')
        versions = {}
        for name in REQUIRED:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        write_json(run / 'ENVIRONMENT.json', {
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': versions, 'expected_packages': REQUIRED,
            'threads_for_study': 1})
        if not args.audit_only and versions != REQUIRED:
            raise RuntimeError('DEPENDENCY_MISMATCH: use the isolated environment and requirements.txt; see ENVIRONMENT.json')
        shutil.copy2(ROOT / 'PROTOCOL.json', run / 'PROTOCOL.json')
        shutil.copytree(ROOT / 'reference' / 'kanids', run / 'source' / 'kanids',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['run_stage2.py', 'study.py', 'audit_dataset.py', 'pilot.py', 'requirements.txt', 'KIT_MANIFEST.json', 'STAGE1_PROTOCOL.json']:
            shutil.copy2(ROOT / name, run / 'source' / name)
        for rel in ['validation/UNIT_TESTS.txt', 'validation/STAGE1_WINDOWS_REVIEW.json']:
            if (ROOT / rel).is_file():
                (run / 'preflight').mkdir(exist_ok=True)
                shutil.copy2(ROOT / rel, run / 'preflight' / Path(rel).name)
        if args.setup_log:
            (run / 'preflight').mkdir(exist_ok=True)
            shutil.copy2(args.setup_log, run / 'preflight' / 'USER_SETUP_AND_TESTS.log')
        capture_project(args.project, run)
        csv_path = ROOT / 'data' / 'train_test_network.csv'
        report = audit(csv_path, run / 'audit')
        record['source_csv_sha256'] = report['source_csv_sha256']
        print('PAIR_SPLIT_AUDIT_PASS', flush=True)
        for name, info in report['splits'].items():
            print(name, 'rows=', info['rows'], 'normal=', info['normal'], 'attack=', info['attack'], flush=True)
        if args.audit_only:
            record['status'] = 'AUDIT_COMPLETE_NO_TRAINING'
        else:
            command = [sys.executable, '-u', str(run / 'source' / 'study.py'),
                       '--csv', str(csv_path), '--split', str(run / 'audit' / 'split_assignments.csv.gz'),
                       '--out', str(run / 'study'), '--source-dir', str(run / 'source'), '--protocol', str(run / 'PROTOCOL.json')]
            write_json(run / 'COMMAND.json', {'argv': command})
            env = os.environ.copy()
            env.update({'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
                        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                        'NUMEXPR_NUM_THREADS': '1', 'KANIDS_ARTIFACTS': str(run / 'source' / 'artifacts')})
            with (run / 'study.log').open('w', encoding='utf-8', newline='\n') as log:
                with subprocess.Popen(command, cwd=run, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                      errors='replace', bufsize=1) as process:
                    try:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            log.flush()
                        returncode = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
            record['worker_exit_code'] = returncode
            if returncode:
                raise RuntimeError('FIVE_SEED_STUDY_FAILED: see study.log')
            worker_status = json.loads((run / 'study' / 'status.json').read_text(encoding='utf-8'))
            if not str(worker_status.get('status', '')).startswith('STUDY_COMPLETE'):
                raise RuntimeError('Worker did not report a completed study: ' + str(worker_status))
            record['nonconverged'] = bool(worker_status.get('nonconverged', False))
            record['status'] = ('STUDY_COMPLETE_WITH_CONVERGENCE_WARNING' if record['nonconverged']
                                else 'STUDY_COMPLETE_FIVE_SEEDS')
            record['test_predictions_computed'] = True
            if record['nonconverged']:
                print('REVIEW_REQUIRED: optimizer reached its fixed budget; see recorded convergence warnings.', flush=True)
        exit_code = 0
    except KeyboardInterrupt:
        record['status'] = 'INTERRUPTED'
        record['error'] = 'Interrupted by user; previous results preserved.'
        print(record['error'], flush=True)
        exit_code = 130
    except Exception as exc:
        record['status'] = 'BLOCKED_OR_FAILED'
        record['error'] = str(exc)
        (run / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8', newline='\n')
        print('STOP:', str(exc), flush=True)
    finally:
        started = (run / 'study' / 'TEST_EVALUATION_STARTED.json').is_file()
        complete = str(record['status']).startswith('STUDY_COMPLETE')
        record['test_evaluation_started'] = started
        record['test_evaluation_complete'] = complete
        record['test_predictions_computed'] = True if complete else (None if started else False)
        if started and not complete:
            record['test_access_note'] = 'Test evaluation started; partial predictions may exist. This run must not be described as test-unseen.'
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(run / 'RUN_RECORD.json', record)
        target = archive_run(run)
        print('\nSTATUS:', record['status'], flush=True)
        print('SEND_THIS_ZIP:', target, flush=True)
        print('ZIP_SHA256:', sha256(target), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
