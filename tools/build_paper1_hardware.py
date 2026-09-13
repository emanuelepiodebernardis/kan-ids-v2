#!/usr/bin/env python3
"""Build one common-cohort target and record evidence. Never uploads or opens serial."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_identity():
    paths = [ROOT / 'mcu_pio/platformio.ini']
    for folder in ['src', 'include']:
        paths.extend(p for p in (ROOT / 'mcu_pio' / folder).rglob('*') if p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--board', choices=['megaatmega2560', 'esp32c3'], required=True)
    ap.add_argument('--mode', choices=['latency', 'energy'], default='latency')
    ap.add_argument('--variant', choices=['coeff', 'lut14', 'mlcoeff', 'mlp', 'dt5'], default='coeff')
    ap.add_argument('--out', type=Path, required=True, help='New evidence folder; existing folders are refused')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    env_name = a.board + '_common_' + ('energy_' if a.mode == 'energy' else '') + a.variant
    before = source_identity()
    command = [sys.executable, '-m', 'platformio', 'run', '-d', str(ROOT / 'mcu_pio'), '-e', env_name]
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'environment': env_name,
              'command': command, 'source_sha256': before, 'uploaded': False,
              'physical_measurements': False, 'status': 'BUILD_STARTED',
              'target_note': 'ESP32-C3 uses inherited devkitm-1 definition; actual board and power setup require confirmation.'}
    record_path = a.out / 'build_record.json'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    version = subprocess.run([sys.executable, '-m', 'platformio', '--version'], capture_output=True, text=True)
    record['platformio_version_output'] = version.stdout + version.stderr
    with (a.out / 'build.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    record['returncode'] = result.returncode
    record['source_unchanged_during_build'] = before == source_identity()
    build_dir = ROOT / 'mcu_pio/.pio/build' / env_name
    products = [build_dir / ('firmware.' + ext) for ext in ['elf', 'hex', 'bin']]
    record['build_products'] = {p.name: {'sha256': sha(p), 'file_bytes': p.stat().st_size,
                                       'path': p.relative_to(ROOT).as_posix()}
                                for p in products if p.is_file()}
    required = ['firmware.elf', 'firmware.hex' if a.board == 'megaatmega2560' else 'firmware.bin']
    ok = result.returncode == 0 and record['source_unchanged_during_build'] and all(p in record['build_products'] for p in required)
    record['status'] = 'BUILD_PASS_NOT_FLASHED' if ok else 'BUILD_FAILED_OR_INCOMPLETE'
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    record['size_note'] = 'file_bytes is file length. Linked Flash/static RAM must be read from build.log; neither is peak runtime RAM.'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(record['status']); print(record_path)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
