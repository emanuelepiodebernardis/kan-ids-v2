#!/usr/bin/env python3
"""Build five frozen C3 variants and coordinate separate, manually saved FNB58 traces."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

MODELS = ('coeff', 'lut', 'mlp', 'kanml', 'dt5')
MODEL_BYTES = dict(coeff=254, lut=20554, mlp=760, kanml=5244, dt5=285)
PROTOCOL = 'kanids-c3-fnb58-suite-v1'
EXPECTED_SERIAL = '10:00:3B:CB:8D:70'
COHORT_SHA = '20c53b571bfaef91f5fd26b19e5960e3fa85ed48b38c78c8527245b921ebe86f'
RAW_IDS = [57978,126454,49019,94298,183174,127803,12386,91098,14299,116321,
           148681,119000,150843,98347,80922,124992,204398,82285,25406,123258]
EVENT_NAMES = ['pre_led_on_1','pre_led_off_1','pre_led_on_2','pre_led_off_2',
               'pre_led_on_3','pre_led_off_3','active_begin','active_end',
               'post_led_on_1','post_led_off_1','post_led_on_2','post_led_off_2',
               'post_led_on_3','post_led_off_3']
BINARY_NAMES = ('firmware.elf', 'firmware.bin', 'partitions.bin', 'bootloader.bin')

class PilotError(RuntimeError):
    pass

def utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

def digest(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def write_json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

def canonical_serial(value):
    return str(value or '').replace(':', '').replace('-', '').upper()

def check_board(ports, requested=None):
    ports = list(ports)
    if requested:
        selected = [p for p in ports if p.device.casefold() == requested.casefold()]
        if len(selected) != 1:
            raise PilotError('Порт ' + requested + ' не найден однозначно.')
        p = selected[0]
        if (p.vid, p.pid) != (0x303A, 0x1001) or canonical_serial(p.serial_number) != canonical_serial(EXPECTED_SERIAL):
            raise PilotError('На ' + requested + ' другая плата: ' + str(p.serial_number) +
                             '. Нужна прежняя ESP32-C3 с SER=' + EXPECTED_SERIAL + '; прошивка не начата.')
    selected = [p for p in ports if (p.vid, p.pid) == (0x303A, 0x1001)
                and canonical_serial(p.serial_number) == canonical_serial(EXPECTED_SERIAL)]
    if len(selected) != 1:
        raise PilotError('Прежняя ESP32-C3 SER=' + EXPECTED_SERIAL + ' не найдена однозначно.')
    p = selected[0]
    return {'port': p.device, 'vid': p.vid, 'pid': p.pid,
            'serial_number': p.serial_number, 'description': p.description}

def rediscover(comports, timeout=45):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return check_board(comports())
        except PilotError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)

def verify_sources(root):
    root = Path(root).resolve()
    entries = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8')).get('files')
    if not isinstance(entries, list) or not entries:
        raise PilotError('Пустой SOURCE_MANIFEST.json')
    seen = set()
    for entry in entries:
        name = entry.get('path', '')
        path = (root / name).resolve()
        if not name or name in seen or not path.is_relative_to(root) or not path.is_file():
            raise PilotError('Неверный путь в манифесте: ' + name)
        if digest(path) != {'bytes': entry.get('bytes'), 'sha256': entry.get('sha256')}:
            raise PilotError('Файл изменён: ' + name)
        seen.add(name)
    required = {'project/platformio.ini', 'project/src/main.cpp', 'run_c3_energy_suite.py', 'cfn_decoder.py'}
    for folder in ('project/include','project/src'):
        required.update(p.relative_to(root).as_posix() for p in (root / folder).rglob('*') if p.is_file())
    if not required.issubset(seen):
        raise PilotError('Исходник отсутствует в манифесте: ' + str(required-seen))
    return entries

def copy_sources(root, destination, entries):
    for entry in entries + [{'path': 'SOURCE_MANIFEST.json'}]:
        src = root / entry['path']
        dst = destination / entry['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def fields(line, tag):
    parts = line.split()
    if not parts or parts[0] != tag:
        raise PilotError('Ожидалось ' + tag + ', получено: ' + line)
    values = {}
    for part in parts[1:]:
        if '=' not in part:
            raise PilotError('Некорректная строка протокола: ' + line)
        key, value = part.split('=', 1)
        if not key or not value or key in values:
            raise PilotError('Повторное/пустое поле: ' + line)
        values[key] = value
    return values

def integer(values, name):
    value = values.get(name, '')
    if not re.fullmatch(r'[0-9]+', value):
        raise PilotError('Нет целого поля ' + name)
    return int(value)

def validate_info(lines, model):
    tags = {}
    wanted = {'HELLO','MODEL','SYSTEM','RAW_IDS','COHORT','WDT','INFO_DONE'}
    for line in lines:
        tag = line.split()[0]
        if tag not in wanted or tag in tags:
            raise PilotError('Неверный/повторный INFO: ' + line)
        tags[tag] = fields(line, tag)
    if set(tags) != wanted or tags['HELLO'] != {'protocol': PROTOCOL}:
        raise PilotError('Неполный INFO или другой протокол')
    if tags['MODEL'] != {'variant':model, 'model_bytes':str(MODEL_BYTES[model]), 'cohort_sha256':COHORT_SHA}:
        raise PilotError('Другая модель/когорта')
    if tags['RAW_IDS'] != {'values': ','.join(map(str, RAW_IDS))}:
        raise PilotError('Другие исходные строки/порядок')
    if tags['COHORT'] != {'rows':'first20','order':'attack_normal_interleaved','boundary':'prepared_features_in_RAM','model_placement':'flash'}:
        raise PilotError('Другая граница измерения')
    system = tags['SYSTEM']
    if canonical_serial(system.get('mac')) != canonical_serial(EXPECTED_SERIAL):
        raise PilotError('MAC прошивки не совпадает с прежней платой')
    for key, value in {'chip':'ESP32-C3','cpu_mhz':'160','timer':'esp_timer_get_time',
                       'timer_bits':'64','led_gpio':'8','led_active':'LOW','interrupts':'enabled',
                       'active_yield':'0','wifi':'not_initialized','bt':'not_initialized'}.items():
        if system.get(key) != value:
            raise PilotError('SYSTEM: не совпадает ' + key)
    if integer(system, 'flash_bytes') != 4194304 or system.get('led_confirmed') not in ('0','1'):
        raise PilotError('SYSTEM: неверный размер Flash или состояние LED')
    wdt = tags['WDT']
    if (wdt.get('policy') != 'idle0_subscription_temporarily_suspended'
            or wdt.get('idle_before') not in ('0','1','disabled')
            or wdt.get('loop_before') not in ('0','disabled')):
        raise PilotError('INFO: неожиданное состояние WDT')
    if tags['INFO_DONE'].get('state') not in ('idle','ready','done'):
        raise PilotError('Плата не готова к ARM')
    return tags

def validate_ready(lines):
    if len(lines) != 2:
        raise PilotError('ARM должен вернуть CORRECTNESS и READY')
    if fields(lines[0], 'CORRECTNESS') != {'checked':'20','correct':'1','reference':'compiled_C_predictions'}:
        raise PilotError('Предсказания на 20 строках не прошли проверку')
    values = fields(lines[1], 'READY')
    ready = {key: integer(values,key) for key in ('count','target_us','cal_n','cal_us','checksum_per20')}
    cal_n = ready['cal_n']
    if not (ready['target_us'] == 120000000 and ready['checksum_per20'] == 10
            and cal_n >= 200 and cal_n <= 6553600 and cal_n % 200 == 0
            and (cal_n//200) & ((cal_n//200)-1) == 0
            and 100000 <= ready['cal_us'] <= 30000000
            and 20 <= ready['count'] <= 2000000000 and ready['count'] % 20 == 0):
        raise PilotError('READY: калибровка вне протокола')
    expected = (ready['target_us'] * cal_n // ready['cal_us'] // 20) * 20
    if ready['count'] != expected:
        raise PilotError('READY: число вызовов не соответствует калибровке')
    return ready

def validate_done(lines, ready):
    done = None
    events = []
    watchdog = []
    prep = terminal = False
    for line in lines:
        if terminal:
            raise PilotError('Данные после WAIT_DONE')
        tag = line.split()[0]
        obj = fields(line, tag)
        if tag == 'PREP':
            if prep or done or events or len(watchdog) != 1 or obj != {'sync':'begin'}:
                raise PilotError('Неверный PREP')
            prep = True
        elif tag == 'WDT':
            if obj.get('phase') == 'before':
                if prep or done or watchdog:
                    raise PilotError('WDT before должен предшествовать PREP')
            elif obj.get('phase') == 'after':
                if done is None or len(events) != 14 or len(watchdog) != 1:
                    raise PilotError('WDT after должен следовать за всеми EVENT')
            else:
                raise PilotError('Неверная фаза WDT')
            watchdog.append(obj)
        elif tag == 'DONE':
            if not prep or done is not None or events:
                raise PilotError('Повторный или преждевременный DONE')
            done = {k:integer(obj,k) for k in ('count','active_us','checksum','expected','correct','timing_ok','wdt_restored')}
        elif tag == 'EVENT':
            if done is None:
                raise PilotError('EVENT до DONE')
            events.append({'name':obj.get('name'),'us':integer(obj,'us')})
        elif tag == 'WAIT_DONE':
            if obj != {'reset_required':'0','rearm_allowed':'1'}:
                raise PilotError('Неверное завершение')
            terminal = True
        else:
            raise PilotError('Ошибка/неожиданное сообщение: ' + line)
    if not done or not terminal:
        raise PilotError('Опыт не завершён')
    expected = ready['count']//20 * ready['checksum_per20']
    if not (done['count']==ready['count'] and done['checksum']==done['expected']==expected
            and done['correct']==done['timing_ok']==done['wdt_restored']==1 and 60000000<=done['active_us']<=180000000):
        raise PilotError('DONE: неверное число, контрольная сумма или длительность')
    if [e['name'] for e in events] != EVENT_NAMES:
        raise PilotError('Неверный порядок событий')
    ts = [e['us'] for e in events]
    if not (0 <= ts[0] <= 100000 and all(b>a for a,b in zip(ts,ts[1:]))):
        raise PilotError('Немонотонные времена событий')
    if ts[7]-ts[6] != done['active_us']:
        raise PilotError('EVENT и active_us расходятся')
    gaps = [2000000,1000000,4000000,1000000,2000000,10000000,done['active_us'],10000000,
            2000000,1000000,4000000,1000000,2000000]
    if any(abs((b-a)-gap)>100000 for a,b,gap in zip(ts,ts[1:],gaps)):
        raise PilotError('Маркер/защитный интервал вне допуска 100 ms')
    if len(watchdog) != 2 or [w.get('phase') for w in watchdog] != ['before','after']:
        raise PilotError('Нет обеих квитанций WDT')
    before, after = watchdog
    if before.get('loop_before') not in ('0','disabled') or before.get('idle_before') not in ('0','1','disabled'):
        raise PilotError('Неожиданная подписка WDT')
    suspended = '1' if before['idle_before']=='1' else '0'
    if (before.get('suspended') != suspended or after.get('restored') != '1'
            or any(before.get(k) != after.get(k) for k in ('idle_before','loop_before','suspended'))):
        raise PilotError('WDT не восстановлен')
    return {'done':done, 'events':events, 'watchdog':watchdog, 'software_protocol_pass':True,
            'external_trace_pending':True, 'energy_result_available':False}

def run_command(command, directory, stem, record):
    env = os.environ.copy()
    env['PYTHONIOENCODING']='utf-8'
    env['PYTHONUTF8']='1'
    info = {'argv':command,'started_utc':utc(),'log':stem+'.log','child_python_io_encoding':'utf-8'}
    record.setdefault('commands',[]).append(info)
    print('\n> ' + subprocess.list2cmdline(command), flush=True)
    with (directory/(stem+'.log')).open('w',encoding='utf-8',newline='\n') as handle:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True,encoding='utf-8',errors='replace',env=env)
        try:
            for line in proc.stdout:
                handle.write(line); handle.flush()
                print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
            info['interrupted']=True
            raise
        finally:
            proc.stdout.close()
            info['finished_utc']=utc()
            info['returncode']=proc.returncode
    if code:
        raise PilotError('Команда завершилась с ошибкой: '+stem)

class SerialCapture:
    def __init__(self, device, directory):
        self.device=device
        self.raw=(directory/'serial_raw.bin').open('wb')
        self.log=(directory/'serial.jsonl').open('w',encoding='utf-8',newline='\n')
        self.text=(directory/'serial.log').open('w',encoding='utf-8',newline='\n')
        self.buffer=bytearray(); self.pending=deque()
    def entry(self,direction,data):
        self.log.write(json.dumps({'utc':utc(),'host_monotonic_ns':time.perf_counter_ns(),
                                  'direction':direction,'hex':data.hex()})+'\n');self.log.flush()
    def line(self,deadline):
        while time.monotonic()<deadline:
            if self.pending:
                line=self.pending.popleft().decode('ascii',errors='strict').rstrip('\r')
                if line:
                    self.text.write(line+'\n');self.text.flush();print(line,flush=True);return line
                continue
            data=self.device.read(min(max(self.device.in_waiting,1),4096))
            if not data: continue
            self.raw.write(data);self.raw.flush();self.entry('RX',data);self.buffer.extend(data)
            while b'\n' in self.buffer:
                value,_,rest=self.buffer.partition(b'\n');self.pending.append(bytes(value));self.buffer=bytearray(rest)
            if len(self.buffer)>4096: raise PilotError('Слишком длинная строка USB')
        raise PilotError('Время ожидания платы истекло; измерение не принято')
    def send(self,command):
        data=(command+'\n').encode('ascii');self.entry('TX',data)
        if self.device.write(data)!=len(data): raise PilotError('Команда передана не полностью')
        self.device.flush()
    def until(self,terminal,timeout=60,boot_noise=False):
        result=[]; deadline=time.monotonic()+timeout; started=not boot_noise
        while True:
            line=self.line(deadline)
            if not started:
                if line.startswith('HELLO '): started=True
                elif line.startswith('ERROR '): raise PilotError(line)
                else: continue
            result.append(line)
            if line.startswith('ERROR '): raise PilotError(line)
            if line.startswith(terminal+' '): return result
    def close(self):
        for handle in (self.raw,self.log,self.text):handle.close()

def make_archive(directory):
    """Write atomically, so a partial update cannot destroy the previous valid ZIP."""
    archive=directory.with_suffix('.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob('*')):
            if path.is_file(): zf.write(path,path.relative_to(directory.parent).as_posix())
    temporary.replace(archive)
    return archive

def cfn_path(value):
    value=value.strip()
    if len(value)>=2 and value[0]==value[-1] and value[0] in '\"\'':value=value[1:-1]
    return Path(value).expanduser()

def validate_cfn(path, active_us, used_hashes):
    from cfn_decoder import decode
    path=Path(path)
    if not path.is_file() or path.suffix.lower()!='.cfn' or path.stat().st_size<26:
        raise PilotError('Нужен существующий непустой файл .cfn')
    identity=digest(path)
    if identity['sha256'] in used_hashes:
        raise PilotError('Этот CFN уже использован в данной серии; сохраните новую запись')
    try: metadata,rows=decode(path)
    except (ValueError,OverflowError) as exc: raise PilotError('CFN не читается полностью: '+str(exc)) from exc
    if not metadata['decoded_to_exact_eof'] or metadata['sample_rate_header_sps']!=10:
        raise PilotError('Требуется полная запись CFN при 10 sps')
    codes={c['code'] for c in metadata['channels']}
    if not {0,1,4}.issubset(codes):raise PilotError('В CFN должны быть VBUS, IBUS, PBUS')
    differences=[b[0]-a[0] for a,b in zip(rows,rows[1:])]
    if any(abs(delta-0.1)>1e-7 for delta in differences):
        raise PilotError('В CFN неверная временная сетка/пропуски')
    span=rows[-1][0]-rows[0][0]
    if span < active_us/1000000+40:
        raise PilotError('CFN слишком короткий: %.1f s; нужны активная серия и оба маркера' % span)
    if metadata['start_current_mA']!=0 or metadata['stop_current_mA']!=0:
        raise PilotError('CFN записан с ненулевыми Start/Stop CUR')
    return {'file':identity,'metadata':metadata,'span_s':span,'mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
            'structural_pass':True,'marker_alignment_pending':True,'energy_result_available':False}

def collect_cfn(directory, name, active_us, used_hashes, prompt=input):
    print('\nВ UsbMeter: квадрат Stop → Сохранить. Имя: '+name)
    print('Полный путь для окна Сохранить: '+str(Path(__file__).resolve().parent/'captures'/name))
    while True:
        value=prompt('Вставьте полный путь к сохранённому CFN (или Enter для ожидаемого пути): ')
        path=cfn_path(value) if value.strip() else Path(__file__).resolve().parent/'captures'/name
        try:
            audit=validate_cfn(path,active_us,used_hashes)
            target=directory/name
            if target.exists():raise PilotError('Целевой CFN уже существует')
            shutil.copy2(path,target)
            if digest(target)!=audit['file']:raise PilotError('CFN изменился при копировании')
            audit['supplied_path']=str(path);audit['saved_name']=name
            write_json(directory/'CFN_RECEIPT.json',audit)
            used_hashes.add(audit['file']['sha256'])
            print('CFN сохранён и структурно проверен. Совмещение маркеров и расчёт энергии будут проверены отдельно.')
            return audit
        except (PilotError,OSError) as exc:
            print('CFN пока не принят: '+str(exc))
            with (directory/'cfn_attempts.jsonl').open('a',encoding='utf-8',newline='\n') as log:
                log.write(json.dumps({'utc':utc(),'path':str(path),'error':str(exc)},ensure_ascii=False)+'\n')
            print('Исправьте путь/сохраните правильную запись. Ctrl+C сохраняет незавершённую серию.')

def copy_binaries(project,model,destination):
    location=project/'.pio/build'/('esp32c3_fnb58_'+model+'_pilot')
    identities={}
    destination.mkdir(parents=True,exist_ok=True)
    for name in BINARY_NAMES:
        src=location/name
        if not src.is_file():raise PilotError('Не найден результат сборки '+str(src))
        identities[name]=digest(src);shutil.copy2(src,destination/name)
    return identities

def preserve_upload_support(project, destination):
    """Resolve the exact pinned Arduino package, including the extra boot_app0 image."""
    from platformio.project.config import ProjectConfig
    from platformio.package.manager.tool import ToolPackageManager
    config=ProjectConfig.get_instance(str(project/'platformio.ini'))
    manager=ToolPackageManager(config.get('platformio','packages_dir'))
    package=manager.get_package('framework-arduinoespressif32@3.20017.241212')
    if package is None:
        raise PilotError('Не найден закреплённый пакет framework-arduinoespressif32')
    package_path=Path(package.path)
    source=package_path/'tools/partitions/boot_app0.bin'
    if not source.is_file():
        raise PilotError('Не найден дополнительный образ загрузки boot_app0.bin')
    identity=digest(source)
    shutil.copy2(source,destination/'boot_app0.bin')
    manifest=package_path/'package.json'
    if manifest.is_file():shutil.copy2(manifest,destination/'framework-package.json')
    return {'source_path':str(source),'file':identity,'flash_offset':'0xe000',
            'package_version':str(package.metadata.version),
            'package_manifest':digest(manifest) if manifest.is_file() else None}


def assert_binaries(project,model,identities):
    location=project/'.pio/build'/('esp32c3_fnb58_'+model+'_pilot')
    for name,identity in identities.items():
        if digest(location/name)!=identity:raise PilotError('Бинарник изменился при upload: '+name)

def open_device(serial_module,board):
    device=serial_module.Serial(port=None,baudrate=115200,timeout=0.25,write_timeout=3)
    device.dtr=True;device.rts=False;device.port=board['port'];device.open()
    return device

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--build-all',action='store_true',help='собрать все пять вариантов без загрузки')
    action.add_argument('--run-all',action='store_true',help='собрать и измерить выбранные варианты')
    parser.add_argument('--port',default='COM10',help='текущий порт прежней C3; после upload ищется по серийному номеру')
    parser.add_argument('--model',choices=MODELS,help='только один вариант')
    parser.add_argument('--repeats',type=int,choices=(1,2),default=2)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parent;project=root/'project'
    suite_id=datetime.now(timezone.utc).strftime('C3_ENERGY_SUITE_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]
    directory=root/'runs'/suite_id;directory.mkdir(parents=True,exist_ok=False)
    (root/'captures').mkdir(exist_ok=True)
    models=(args.model,) if args.model else MODELS
    record={'suite_id':suite_id,'started_utc':utc(),'status':'incomplete','models':list(models),
            'repeats':args.repeats,'mode':'run_all' if args.run_all else 'build_all',
            'python':sys.version,'system':platform.platform(),'requested_port':args.port,
            'expected_serial':EXPECTED_SERIAL,'commands':[],'builds':{},'runs':[],
            'measurement_boundary':'prepared_features_in_RAM; whole-board USB input including downstream cable',
            'host_timestamps':'transport receipts only, hardware event times are firmware esp_timer 64-bit values',
            'energy_result_available':False}
    current=None;device=capture=None;exit_code=1;used_hashes=set()
    try:
        if sys.version_info[:2]!=(3,11):raise PilotError('Запустите через py -3.11')
        entries=verify_sources(root);copy_sources(root,directory/'sources',entries)
        record['source_manifest']=digest(root/'SOURCE_MANIFEST.json')
        pio_version=importlib.metadata.version('platformio')
        if pio_version!='6.1.19':raise PilotError('Требуется PlatformIO 6.1.19, установлен '+pio_version)
        record['versions']={'platformio':pio_version,'pyserial':importlib.metadata.version('pyserial')}
        import serial
        from serial.tools import list_ports
        if args.run_all:record['board_initial']=check_board(list_ports.comports(),args.port)
        base=[sys.executable,'-m','platformio']
        for model in models:
            env='esp32c3_fnb58_'+model+'_pilot';out=directory/'builds'/model;out.mkdir(parents=True)
            build={'model':model,'environment':env,'commands':[]};record['builds'][model]=build
            run_command(base+['run','-d',str(project),'-e',env],out,'01_build',build)
            run_command(base+['pkg','list','-d',str(project),'-e',env],out,'02_packages',build)
            verify_sources(root);build['binaries']=copy_binaries(project,model,out/'binaries')
            build['upload_support']=preserve_upload_support(project,out/'binaries')
            write_json(out/'BUILD_RECORD.json',build)
        if args.build_all:
            record['status']='builds_pass_no_upload';exit_code=0
        else:
            led_confirmed=False
            for model in models:
                env='esp32c3_fnb58_'+model+'_pilot';setup=directory/'setup'/model;setup.mkdir(parents=True)
                setup_record={'commands':[],'board_before_upload':check_board(list_ports.comports())}
                record.setdefault('setups',{})[model]=setup_record
                verify_sources(root)
                run_command(base+['run','-d',str(project),'-e',env,'-t','upload','--upload-port',setup_record['board_before_upload']['port']],setup,'03_upload',setup_record)
                verify_sources(root);assert_binaries(project,model,record['builds'][model]['binaries'])
                support=record['builds'][model]['upload_support']
                if digest(Path(support['source_path']))!=support['file']:
                    raise PilotError('boot_app0.bin изменился во время upload')
                board=rediscover(list_ports.comports);setup_record['board_after_upload']=board
                device=open_device(serial,board);capture=SerialCapture(device,setup)
                # INFO is explicit: startup banners are not required and native USB may enumerate late.
                time.sleep(1);capture.send('INFO');info=validate_info(capture.until('INFO_DONE',60,boot_noise=True),model)
                setup_record['info']=info
                if not led_confirmed:
                    print('\nПроверка светодиода: смотрите на C3. Ожидаются вспышки 2, 4 и 2 секунды с паузами.')
                    capture.send('LEDTEST');led_lines=capture.until('LEDTEST',30)
                    if led_lines!=['LEDTEST state=begin']:raise PilotError('Нет начала LEDTEST')
                    if capture.until('LEDTEST',30)!=['LEDTEST state=done']:raise PilotError('Нет конца LEDTEST')
                    answer=input('Увидели три вспышки именно управляемого LED (2–4–2 s)? Введите ДА: ').strip().casefold()
                    if answer not in ('да','yes','y','1'):raise PilotError('LED не подтверждён; опыты не начаты')
                    led_confirmed=True
                    record['led_observation']={'confirmed':True,'operator_answer':answer,'utc':utc(),'setup_model':model,'gpio':8,'active':'LOW'}
                capture.send('LEDCONFIRM')
                if capture.until('LEDCONFIRM',10)!=['LEDCONFIRM confirmed=1']:raise PilotError('Прошивка не подтвердила LED')
                capture.close();capture=None;write_json(setup/'SETUP_RECORD.json',setup_record)
                for repeat in range(1,args.repeats+1):
                    run_id=f'C3_{model.upper()}_{repeat:02d}_'+uuid.uuid4().hex[:8]
                    current_dir=directory/'physical_runs'/run_id;current_dir.mkdir(parents=True)
                    current={'run_id':run_id,'suite_id':suite_id,'model':model,'repeat':repeat,'started_utc':utc(),
                             'status':'incomplete','board':check_board(list_ports.comports()),
                             'source_manifest':record['source_manifest'],'binaries':record['builds'][model]['binaries'],
                             'setup_path':setup.relative_to(directory).as_posix(),
                             'build_path':f'builds/{model}','sources_path':'sources','energy_result_available':False}
                    record['runs'].append(current);capture=SerialCapture(device,current_dir)
                    capture.send('INFO');current['info']=validate_info(capture.until('INFO_DONE',30),model)
                    capture.send('ARM');ready=validate_ready(capture.until('READY',120));current['ready']=ready
                    name=f'c3_{model}_pilot_{repeat:02d}_{suite_id.rsplit("_",1)[-1]}.cfn'
                    print('\n=== '+model+' / повтор '+str(repeat)+' из '+str(args.repeats)+' ===')
                    print('В UsbMeter: 10 sps, Start CUR=0, Stop CUR=0; VBUS/IBUS/PBUS включены.')
                    print('Файл для сохранения после опыта: '+str(root/'captures'/name))
                    print('Создайте НОВУЮ запись (лист), нажмите ▶. Проверьте, что TIME действительно растёт.')
                    input('Когда TIME растёт, нажмите Enter здесь: ')
                    current['run_command_utc']=utc();capture.send('RUN')
                    print('Опыт около 3 минут. Не трогайте Reset, питание и настройки. Ждите сообщения о сохранении CFN.')
                    result_lines=capture.until('WAIT_DONE',600);current['result']=validate_done(result_lines,ready)
                    current['status']='software_protocol_pass_external_trace_pending'
                    capture.close();capture=None
                    current['cfn']=collect_cfn(current_dir,name,current['result']['done']['active_us'],used_hashes)
                    current['status']='software_and_cfn_structure_pass_alignment_pending';current['finished_utc']=utc()
                    write_json(current_dir/'RUN_RECORD.json',current);make_archive(current_dir)
                    current=None;write_json(directory/'SUITE_RECORD.json',record);make_archive(directory)
                    print('Повтор сохранён. Следующий начнётся только после новой записи FNB58.')
                device.close();device=None
            record['status']='all_runs_saved_alignment_and_energy_pending';exit_code=0
    except KeyboardInterrupt:
        record['error']='interrupted_by_user'
        print('\nОстановлено. Начатая серия на плате может ещё выполняться. Сохраните CFN; архив сохраняется.',file=sys.stderr)
    except Exception as exc:
        record['error']=type(exc).__name__+': '+str(exc)
        print('\nОСТАНОВКА: '+str(exc)+'\nСледующая загрузка/серия не выполняется.',file=sys.stderr)
    finally:
        if capture:capture.close()
        if device:device.close()
        if current is not None:
            current['error']=record.get('error','incomplete');current['finished_utc']=utc()
            write_json(current_dir/'RUN_RECORD.json',current)
            make_archive(current_dir)
        record['finished_utc']=utc();write_json(directory/'SUITE_RECORD.json',record)
        archive=make_archive(directory)
        print('\nАрхив всей серии: '+str(archive))
        print('Пришлите этот ZIP. CFN, успешно принятые программой, уже внутри.')
    return exit_code

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(errors='replace');sys.stderr.reconfigure(errors='replace')
    raise SystemExit(main())
