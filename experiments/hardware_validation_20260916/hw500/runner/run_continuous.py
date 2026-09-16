#!/usr/bin/env python3
"""One uninterrupted FNB58 recording per identity-locked 26-run board series."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

from run_suite import (ACCEPTED, BOARDS, MODELS, GateError, SerialCapture, assert_binaries,
    check_board, check_versions, copy_binaries, copy_sources, digest, env_name,
    make_archive, open_device, plan_for, read_json, rediscover, run_command,
    utc, validate_done, validate_info, validate_ready, verify_packet,
    verify_sources, write_json)

ACQUISITION = 'kanids-hw500-continuous-v1'
PENDING = 'software_pass_trace_pending'
ANALYSIS_ACCEPTED = 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_SERIES'


class ContinuousCapture(SerialCapture):
    """Record completion of each line at its final RX chunk, not console-print time."""
    def __init__(self, device, directory):
        super().__init__(device, directory)
        self.protocol_timing = {}
        self.run_active = False

    def entry(self, direction, data):
        stamp = time.perf_counter_ns()
        self.log.write(json.dumps({'utc': utc(), 'host_monotonic_ns': stamp,
                                  'direction': direction, 'hex': data.hex()}) + '\n')
        self.log.flush()
        if direction == 'TX' and data == b'RUN\n':
            self.run_active = True
            self.protocol_timing = {'run_send_monotonic_ns': stamp}
        return stamp

    def line(self, deadline):
        while time.monotonic() < deadline:
            if self.pending:
                raw, stamp = self.pending.popleft()
                line = raw.decode('ascii', errors='strict').rstrip('\r')
                if not line:
                    continue
                if self.run_active:
                    for tag, name in [('PREP ', 'prep_receive_monotonic_ns'),
                                      ('DONE ', 'done_receive_monotonic_ns')]:
                        if line.startswith(tag):
                            if name in self.protocol_timing:
                                raise GateError('Повторная временная метка ' + tag)
                            self.protocol_timing[name] = stamp
                self.text.write(line + '\n'); self.text.flush()
                print(line, flush=True)
                return line
            data = self.device.read(min(max(self.device.in_waiting, 1), 4096))
            if not data:
                continue
            self.raw.write(data); self.raw.flush()
            stamp = self.entry('RX', data)
            self.buffer.extend(data)
            while b'\n' in self.buffer:
                value, _, rest = self.buffer.partition(b'\n')
                self.pending.append((bytes(value), stamp))
                self.buffer = bytearray(rest)
            if len(self.buffer) > 8192:
                raise GateError('Слишком длинная строка USB')
        raise GateError('Время ожидания платы истекло')


def validate_host_timing(timing):
    keys = ('recording_confirm_monotonic_ns', 'run_send_monotonic_ns',
            'prep_receive_monotonic_ns', 'done_receive_monotonic_ns')
    if any(type(timing.get(k)) is not int or timing[k] <= 0 for k in keys):
        raise GateError('Неполные host monotonic timestamps')
    confirmed, sent, prep, done = [timing[k] for k in keys]
    if not (confirmed <= sent <= prep < done and prep - sent <= 2_000_000_000):
        raise GateError('Нельзя однозначно привязать RUN/PREP к часам компьютера')


def seal_attempt(directory, session, attempt, current):
    """Snapshot completed software evidence; accepted energy is a later decision."""
    attempt = Path(attempt)
    if current['status'] == PENDING:
        validate_host_timing(current['host_timing'])
        snapshot = attempt / 'SOFTWARE_RECORD.json'
        if snapshot.exists():
            raise GateError('Запрещена перезапись исходной квитанции опыта')
        write_json(snapshot, current)
        payload = [{'path': p.relative_to(attempt).as_posix(), **digest(p)}
                   for p in sorted(attempt.rglob('*'))
                   if p.is_file() and p.name not in ('RUN_RECORD.json', 'PACKET_MANIFEST.json')]
        write_json(attempt / 'SOFTWARE_MANIFEST.json', {'schema': 'kanids-hw500-software-evidence-v1', 'files': payload})
    write_json(attempt / 'RUN_RECORD.json', current)
    make_archive(attempt)
    entry = {'relative_path': attempt.relative_to(directory).as_posix(),
             'status': current['status'], 'record_file': digest(attempt / 'RUN_RECORD.json')}
    if current['status'] == PENDING:
        entry['software_record_file'] = digest(attempt / 'SOFTWARE_RECORD.json')
        entry['software_manifest_file'] = digest(attempt / 'SOFTWARE_MANIFEST.json')
    session['attempts'].append(entry)
    write_json(directory / 'SUITE_RECORD.json', session)


def load_pending_records(directory):
    """Verify original UART/software evidence before any trace is associated."""
    directory = Path(directory).resolve()
    session = read_json(directory / 'SUITE_RECORD.json')
    if session.get('schema') != 'kanids-hw500-suite-v1' or session.get('acquisition_protocol') != ACQUISITION:
        raise GateError('Нужна сессия непрерывной записи HW500')
    if session.get('plan') != plan_for(session['board_kind']):
        raise GateError('Изменён план серии')
    actual = {p.relative_to(directory).as_posix() for p in (directory / 'attempts').iterdir() if p.is_dir()}
    listed = [e['relative_path'] for e in session['attempts']]
    if len(set(listed)) != len(listed) or actual != set(listed):
        raise GateError('Незарегистрированная/повторная попытка')
    records, paths, seen = [], [], set()
    for entry in session['attempts']:
        attempt = (directory / entry['relative_path']).resolve()
        if not attempt.is_relative_to(directory):
            raise GateError('Путь опыта вне сессии')
        verify_packet(attempt)
        if digest(attempt / 'RUN_RECORD.json') != entry['record_file']:
            raise GateError('Квитанция RUN_RECORD изменена')
        current = read_json(attempt / 'RUN_RECORD.json')
        if current.get('status') == ACCEPTED:
            raise GateError('Сессия уже содержит принятые результаты; перезапись запрещена')
        if current.get('status') != PENDING:
            continue
        if entry['status'] != PENDING:
            raise GateError('Статус опыта не совпадает с сессией')
        for name, key in [('SOFTWARE_RECORD.json', 'software_record_file'), ('SOFTWARE_MANIFEST.json', 'software_manifest_file')]:
            if digest(attempt / name) != entry.get(key):
                raise GateError('Исходная software-квитанция изменена: ' + name)
        original = read_json(attempt / 'SOFTWARE_RECORD.json')
        if original != current:
            raise GateError('RUN_RECORD отличается от исходной software-квитанции')
        manifest = read_json(attempt / 'SOFTWARE_MANIFEST.json')
        for evidence in manifest['files']:
            path = (attempt / evidence['path']).resolve()
            if not path.is_relative_to(attempt) or digest(path) != {k: evidence[k] for k in ('bytes', 'sha256')}:
                raise GateError('Исходные software/serial данные изменены')
        item = original['plan_item']
        if item not in session['plan'] or item['key'] in seen:
            raise GateError('Неизвестный/повторный опыт')
        if original.get('board_kind') != session['board_kind'] or original.get('suite_id') != session['suite_id']:
            raise GateError('Опыт из другой серии/платы')
        validate_host_timing(original['host_timing'])
        seen.add(item['key']); records.append(original); paths.append(attempt)
    if not records:
        raise GateError('Нет завершённых software-проверок для анализа CFN')
    order = sorted(range(len(records)), key=lambda i: records[i]['plan_item']['index'])
    return session, [records[i] for i in order], [paths[i] for i in order]


def preserve_trace(directory, source):
    """Copy the complete original once, including traces later rejected by analysis."""
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.casefold() != '.cfn':
        raise GateError('Нужен существующий .cfn')
    identity = digest(source)
    target = Path(directory) / 'captures' / ('whole_series_' + identity['sha256'] + '.cfn')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != identity:
            raise GateError('Сохранённая сырая запись изменена')
    else:
        shutil.copyfile(source, target)
    if digest(target) != identity or digest(source) != identity:
        raise GateError('CFN изменился при копировании')
    return target, {'file': identity, 'path': target.relative_to(directory).as_posix(),
                    'session_relative_path': target.relative_to(directory).as_posix(),
                    'original_path': str(source), 'original_mtime_ns': source.stat().st_mtime_ns,
                    'preserved_utc': utc()}


def analyze_saved_session(directory, cfn_path, analyzer=None, enforce_fresh=False):
    """Analyze all completed runs together; never select a favorable subset."""
    directory = Path(directory).resolve()
    session, originals, paths = load_pending_records(directory)
    target, receipt = preserve_trace(directory, cfn_path)
    analysis_id = datetime.now(timezone.utc).strftime('ANALYSIS_%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
    analysis_dir = directory / 'analysis_attempts' / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=False)
    evidence = {'schema': 'kanids-hw500-continuous-analysis-attempt-v1', 'analysis_id': analysis_id,
                'started_utc': utc(), 'status': 'in_progress', 'cfn': receipt,
                'input_records': [{'run_id': r['run_id'], 'software_file': digest(p / 'SOFTWARE_RECORD.json')}
                                  for r, p in zip(originals, paths)]}
    session.setdefault('analysis_attempts', []).append({'path': analysis_dir.relative_to(directory).as_posix(),
                                                       'status': 'in_progress', 'cfn': receipt})
    entry = session['analysis_attempts'][-1]
    write_json(analysis_dir / 'ANALYSIS_RECORD.json', evidence)
    write_json(directory / 'SUITE_RECORD.json', session)
    try:
        if enforce_fresh:
            start = datetime.fromisoformat(session['recording']['confirmed_utc']).timestamp()
            if receipt['original_mtime_ns'] / 1e9 < start - 2:
                raise GateError('CFN сохранён до начала текущей непрерывной записи')
        if analyzer is None:
            from analyze_continuous import analyze_session_trace
            analyzer = analyze_session_trace
        result = analyzer(target, copy.deepcopy(originals))
        analyses = result.get('analyses', [])
        if result.get('status') != ANALYSIS_ACCEPTED or len(analyses) != len(originals):
            raise GateError('Неполный/непринятый анализ непрерывной записи')
        by_id = {a.get('run_id'): a for a in analyses}
        if len(by_id) != len(originals) or set(by_id) != {r['run_id'] for r in originals}:
            raise GateError('Неоднозначное сопоставление результатов с опытами')
        staged = []
        for original, attempt in zip(originals, paths):
            a = by_id[original['run_id']]
            if not (a.get('status') == 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_ESTIMATE'
                    and a.get('board_kind') == original['board_kind'] and a.get('model') == original['model']
                    and a.get('source_sha256') == receipt['file']['sha256']
                    and a.get('workload', {}).get('completed_inferences') == original['result']['done']['count']):
                raise GateError('Неверная идентичность/число вызовов результата анализа')
            if (attempt / 'TRACE_ANALYSIS.json').exists():
                raise GateError('TRACE_ANALYSIS уже существует; требуется проверка прерванной фиксации')
            current = copy.deepcopy(original)
            current.update(status=ACCEPTED, finished_utc=utc(), cfn=receipt,
                           continuous_analysis_path=analysis_dir.relative_to(directory).as_posix())
            staged.append((attempt, current, a))
        # All scientific gates pass before any run becomes accepted.
        write_json(analysis_dir / 'SERIES_ANALYSIS.json', result)
        evidence.update(status='all_completed_software_runs_trace_accepted', finished_utc=utc())
        write_json(analysis_dir / 'ANALYSIS_RECORD.json', evidence)
        for attempt, current, analysis in staged:
            write_json(attempt / 'TRACE_ANALYSIS.json', analysis)
            write_json(attempt / 'CFN_RECEIPT.json', receipt)
            write_json(attempt / 'RUN_RECORD.json', current)
            make_archive(attempt)
            old = next(e for e in session['attempts'] if e['relative_path'] == attempt.relative_to(directory).as_posix())
            old.update(status=ACCEPTED, record_file=digest(attempt / 'RUN_RECORD.json'))
        entry['status'] = evidence['status']
        session['accepted_count'] = len(staged)
        session.pop('error', None)
        full = len(staged) == len(session['plan']) and {r['plan_item']['key'] for r in originals} == {p['key'] for p in session['plan']}
        session['status'] = 'all_26_runs_accepted_descriptive_energy' if full else 'partial_series_trace_accepted_not_complete_campaign'
        session['energy_trace_gate_deferred_until_series_end'] = True
        write_json(directory / 'SUITE_RECORD.json', session)
        if full:
            from summarize_results import summarize_session
            session['summary_status'] = summarize_session(directory)['status']
            write_json(directory / 'SUITE_RECORD.json', session)
        return session
    except BaseException as exc:
        evidence.update(status='failed_or_interrupted_analysis', error=type(exc).__name__ + ': ' + str(exc), finished_utc=utc())
        write_json(analysis_dir / 'ANALYSIS_RECORD.json', evidence)
        entry.update(status=evidence['status'], error=evidence['error'])
        session['error'] = evidence['error']
        write_json(directory / 'SUITE_RECORD.json', session)
        raise
    finally:
        make_archive(analysis_dir)
        make_archive(directory)


def request_saved_trace(capture_path, prompt=None, optional=False):
    prompt = input if prompt is None else prompt
    print('\nFNB58: Stop (квадрат) → Сохранить. Сохраните всю непрерывную запись сюда:\n' + str(capture_path))
    if optional:
        answer = prompt('После сохранения Enter; S = пропустить и сохранить позже: ').strip().casefold()
        if answer == 's':
            return None
    else:
        prompt('После сохранения нажмите Enter: ')
    if not Path(capture_path).is_file():
        raise GateError('CFN не найден: ' + str(capture_path))
    return Path(capture_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', choices=tuple(BOARDS)); parser.add_argument('--port')
    parser.add_argument('--bootstrap-evidence', type=Path)
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--analyze-only', type=Path); parser.add_argument('--cfn', type=Path)
    args = parser.parse_args(argv)
    if args.analyze_only:
        if not args.cfn or args.build_only or args.dry_run:
            parser.error('--analyze-only SESSION требует --cfn FILE и не запускает сборку/плату')
    elif not args.board or args.cfn:
        parser.error('Для записи нужен --board; --cfn используется с --analyze-only')
    if args.dry_run:
        print(json.dumps({'board': BOARDS[args.board], 'plan': plan_for(args.board), 'hardware_access': False,
                          'acquisition_protocol': ACQUISITION, 'recordings_per_board': 1,
                          'trace_gate': 'deferred_until_whole_recording_saved'}, ensure_ascii=False, indent=2))
        return 0
    root = Path(__file__).resolve().parent; project = root / 'project'
    session = directory = current = attempt = device = capture = None
    recording = False; analysis_requested = False; code = 1; capture_path = None
    try:
        entries = verify_sources(root); versions = check_versions()
        if args.analyze_only:
            directory = args.analyze_only.expanduser().resolve()
            verify_packet(directory)
            previous = read_json(directory / 'SUITE_RECORD.json')
            if args.board and args.board != previous['board_kind']:
                raise GateError('Плата CLI не совпадает с сессией')
            if previous['source_manifest'] != digest(root / 'KIT_MANIFEST.json') or previous['versions'] != versions:
                raise GateError('Для анализа нужен тот же пакет и окружение')
            verify_sources(directory / 'sources')
            analysis_requested = True
            session = previous
            session = analyze_saved_session(directory, args.cfn)
            code = 0 if session['status'] == 'all_26_runs_accepted_descriptive_energy' else 2
            return code
        import serial
        from serial.tools import list_ports
        import numpy, scipy
        kind = args.board
        sid = datetime.now(timezone.utc).strftime(kind.upper() + '_HW500_CONTINUOUS_%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
        directory = root / 'runs' / sid; directory.mkdir(parents=True, exist_ok=False)
        session = {'schema': 'kanids-hw500-suite-v1', 'suite_id': sid, 'board_kind': kind,
                   'acquisition_protocol': ACQUISITION, 'expected_board': BOARDS[kind], 'started_utc': utc(),
                   'status': 'incomplete', 'source_manifest': digest(root / 'KIT_MANIFEST.json'),
                   'protocol_file': digest(root / 'PROTOCOL.json'), 'versions': versions,
                   'python': sys.version, 'system': platform.platform(), 'plan': plan_for(kind),
                   'builds': {}, 'attempts': [], 'accepted_count': 0,
                   'measurement_boundary': 'flash_row_load_predict_checksum; whole-board USB supply',
                   'energy_status': 'descriptive_USB_input_estimate_not_calibrated_metrology',
                   'energy_trace_gate_deferred_until_series_end': True,
                   'pilot_gate_before_campaign': 'software_only; meter trace checked after entire recording',
                   'meter': {'model': 'FNIRSI FNB58', 'serial_observed': '104332',
                             'connection': 'operator_confirmed_unchanged_from_previous_campaign',
                             'record_grid_sps': 10, 'start_current_mA': 0, 'stop_current_mA': 0, 'stop_time_s': 5}}
        copy_sources(root, directory / 'sources', entries)
        if args.bootstrap_evidence:
            evidence = args.bootstrap_evidence.resolve(); verify_packet(evidence)
            destination = directory / 'bootstrap' / evidence.name
            shutil.copytree(evidence, destination)
            session['bootstrap_receipts'] = [destination.relative_to(directory).as_posix()]
        captures = root / 'captures' / sid; captures.mkdir(parents=True)
        capture_path = captures / (kind + '_hw500_all_26.cfn')
        session['requested_cfn_path'] = str(capture_path)
        if not args.build_only:
            session['board_observed'] = check_board(list_ports.comports(), kind, args.port)
        write_json(directory / 'SUITE_RECORD.json', session)
        base = [sys.executable, '-m', 'platformio']
        for model in MODELS:
            destination = directory / 'builds' / model; destination.mkdir(parents=True)
            build = {'model': model, 'environment': env_name(kind, model), 'commands': []}
            run_command(base + ['run', '-d', str(project), '-e', env_name(kind, model)], destination, '01_build', build)
            run_command(base + ['pkg', 'list', '-d', str(project), '-e', env_name(kind, model)], destination, '02_packages', build)
            verify_sources(root)
            build['binaries'] = copy_binaries(project, kind, model, destination / 'binaries')
            write_json(destination / 'BUILD_RECORD.json', build)
            session['builds'][model] = build; write_json(directory / 'SUITE_RECORD.json', session)
        if args.build_only:
            session['status'] = 'builds_pass_no_upload'; code = 0; return code
        print('Схема питания прежняя. Версии FNB58/приложения, если видны; Enter = неизвестны.')
        session['meter_versions_operator'] = input('Версии (необязательно): ').strip() or 'unknown_not_reported'
        session['status'] = 'in_progress'
        led_confirmed = False; anchor = None
        for item in session['plan']:
            if item['phase'] == 'campaign' and not any(e['status'] == PENDING for e in session['attempts']):
                raise GateError('Software-пилот не завершён')
            aid = item['key'] + '_' + uuid.uuid4().hex[:8]
            attempt = directory / 'attempts' / aid; attempt.mkdir(parents=True, exist_ok=False)
            current = {'schema': 'kanids-hw500-run-v1', 'run_id': aid, 'suite_id': sid,
                       'board_kind': kind, 'model': item['model'], 'phase': item['phase'], 'repeat': item['repeat'],
                       'cohort_rows': 500, 'plan_item': item, 'started_utc': utc(), 'status': 'incomplete',
                       'acquisition_protocol': ACQUISITION, 'commands': [], 'source_manifest': session['source_manifest'],
                       'protocol_file': session['protocol_file'], 'binaries': session['builds'][item['model']]['binaries'],
                       'build_path': 'builds/' + item['model'], 'sources_path': 'sources'}
            write_json(attempt / 'RUN_RECORD.json', current)
            board = check_board(list_ports.comports(), kind); current['board_before_upload'] = board
            verify_sources(root); assert_binaries(project, kind, item['model'], current['binaries'])
            run_command(base + ['run', '-d', str(project), '-e', env_name(kind, item['model']), '-t', 'nobuild', '-t', 'upload', '--upload-port', board['port']], attempt, '03_upload', current)
            verify_sources(root); assert_binaries(project, kind, item['model'], current['binaries'])
            board = rediscover(list_ports.comports, kind); current['board_after_upload'] = board
            device = open_device(serial, board); capture = ContinuousCapture(device, attempt)
            time.sleep(2); capture.send('INFO')
            current['info'] = validate_info(capture.until('INFO_DONE', 60, boot_noise=True), kind, item['model'])
            print('\n=== ' + kind.upper() + ' ' + str(item['index'] + 1) + '/26: ' + item['key'] + ' ===')
            print('Автоматическая проверка LED: 2–4–2 s; между вспышками 2 s.')
            capture.send('LEDTEST')
            if capture.until('LEDTEST', 30) != ['LEDTEST state=begin'] or capture.until('LEDTEST', 30) != ['LEDTEST state=done']:
                raise GateError('LEDTEST не завершён')
            if not led_confirmed:
                answer = input('Видели три вспышки? Введите ДА: ').strip().casefold()
                if answer not in ('да', 'yes'):
                    raise GateError('Физический LED не подтверждён')
                led_confirmed = True
                session['led_observation'] = {'confirmed': True, 'utc': utc(), 'answer': answer, 'attempt': aid}
            capture.send('CONFIRM_LED')
            if capture.until('LEDCONFIRM', 10) != ['LEDCONFIRM confirmed=1']:
                raise GateError('Нет LEDCONFIRM')
            capture.send('ARM'); current['ready'] = validate_ready(capture.until('READY', 120), item['model'])
            current['stabilization'] = {'started_utc': utc(), 'requested_seconds': 30}
            print('Автоматическая выдержка 30 s. ' + ('Запись FNB58 продолжается.' if recording else 'Запись пока не начинайте.'))
            quiet_start = time.monotonic(); time.sleep(30)
            current['stabilization'].update(finished_utc=utc(), host_elapsed_seconds=time.monotonic() - quiet_start)
            if not recording:
                print('\nFNB58: 10 sps, Start CUR=0, Stop CUR=0, Stop time=5; VBUS/IBUS/PBUS включены.')
                print('Новая запись (лист) → ▶. Проверьте рост TIME и нажмите Enter в течение 10 секунд после ▶.')
                print('Запись одна на все 26 опытов. Не нажимайте паузу/Stop до финального запроса.')
                answer = input('TIME растёт; Enter для всей серии, Q = выход: ').strip().casefold()
                if answer == 'q':
                    raise KeyboardInterrupt
                anchor = time.perf_counter_ns(); recording = True
                session['recording'] = {'confirmed_utc': utc(), 'recording_confirm_monotonic_ns': anchor,
                                        'operator_confirmed_fresh_trace_and_TIME_growth': True,
                                        'operator_max_start_to_confirmation_s': 10}
                write_json(directory / 'SUITE_RECORD.json', session)
            current['run_command_utc'] = utc()
            current['host_timing'] = {'recording_confirm_monotonic_ns': anchor}
            write_json(attempt / 'RUN_RECORD.json', current)
            capture.send('RUN')
            print('Около 3 минут; запись не останавливайте. После опыта следующий начнётся автоматически.')
            current['result'] = validate_done(capture.until('WAIT_DONE', 360), current['ready'], kind)
            current['host_timing'].update(capture.protocol_timing)
            validate_host_timing(current['host_timing'])
            capture.close(); capture = None; device.close(); device = None
            current['status'] = PENDING; current['software_finished_utc'] = utc()
            seal_attempt(directory, session, attempt, current)
            current = attempt = None
            print('Software-проверка пройдена: ' + str(len(session['attempts'])) + '/26. Проверка энергии/маркеров ожидает общий CFN.')
            if item['phase'] == 'pilot':
                print('Пилот прошёл только software-проверку. Пригодность записи проверим по завершении всей серии.')
        session['status'] = 'all_26_software_pass_trace_pending'
        write_json(directory / 'SUITE_RECORD.json', session)
        analysis_requested = True
        source = request_saved_trace(capture_path)
        session = analyze_saved_session(directory, source, enforce_fresh=True)
        code = 0
    except KeyboardInterrupt:
        if session is not None:
            session['error'] = 'interrupted_by_operator'
        print('\nОстановлено. Плата может ещё выполнять текущий RUN. Сохраните непрерывную запись FNB58.', file=sys.stderr)
    except Exception as exc:
        if session is not None:
            session['error'] = type(exc).__name__ + ': ' + str(exc)
        print('\nОСТАНОВКА: ' + str(exc) + '\nСледующая прошивка/опыт не запускается.', file=sys.stderr)
    finally:
        if capture:
            # Even interrupted RUN timing and its partial UART stream are retained.
            if current is not None:
                current.setdefault('host_timing', {}).update(capture.protocol_timing)
            capture.close()
        if device:
            device.close()
        if directory is not None and session is not None:
            if current is not None and attempt is not None:
                current['error'] = session.get('error', 'incomplete'); current['finished_utc'] = utc()
                seal_attempt(directory, session, attempt, current)
            # Reload changes made by analysis even when that call raised.
            if analysis_requested and (directory / 'SUITE_RECORD.json').is_file():
                saved = read_json(directory / 'SUITE_RECORD.json')
                if session.get('error'):
                    saved['error'] = session['error']
                session = saved
            if code and recording and not analysis_requested:
                write_json(directory / 'SUITE_RECORD.json', session)
                try:
                    source = request_saved_trace(capture_path, optional=True)
                    if source is not None:
                        _, receipt = preserve_trace(directory, source)
                        session['partial_cfn_preserved'] = receipt
                except (Exception, KeyboardInterrupt) as exc:
                    session['partial_trace_save_error'] = type(exc).__name__ + ': ' + str(exc)
            session['updated_utc'] = utc()
            session['accepted_count'] = sum(e['status'] == ACCEPTED for e in session['attempts'])
            write_json(directory / 'SUITE_RECORD.json', session)
            archive = make_archive(directory)
            print('\nSEND_THIS_ZIP: ' + str(archive))
            if code:
                if any(e['status'] == PENDING for e in session['attempts']) and session['accepted_count'] == 0:
                    print('Сохранённые software-данные можно разобрать без платы:')
                    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
                    print('& ' + quote(sys.executable) + ' ' + quote(root / 'run_continuous.py') + ' --analyze-only ' + quote(directory) + ' --cfn ' + quote(capture_path or args.cfn))
                elif session['status'] == 'partial_series_trace_accepted_not_complete_campaign':
                    print('Частичная запись разобрана. Полная кампания и итоговая таблица 25 опытов не получены.')
                print('Продолжение измерений после прерывания — новая сессия и новая запись; частичные данные сохраняются.')
    return code


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace'); sys.stderr.reconfigure(errors='replace')
    raise SystemExit(main())
