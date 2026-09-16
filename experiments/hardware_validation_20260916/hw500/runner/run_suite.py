#!/usr/bin/env python3
"""Identity-locked, resumable acquisition of a frozen 500-flow hardware cohort."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
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
EXPECTED_SUMS = dict(coeff=256, lut=256, mlp=254, kanml=250, dt5=249)
PROTOCOL = 'kanids-hw500-v1'
COHORT_SHA = '20c53b571bfaef91f5fd26b19e5960e3fa85ed48b38c78c8527245b921ebe86f'
BOARDS = {'mega': {'serial': '14532303532351804271', 'vid': 0x2341, 'pid': 0x0042, 'chip': 'ATmega2560', 'cpu_mhz': '16'},
          'c3': {'serial': '10:00:3B:CB:8D:70', 'vid': 0x303A, 'pid': 0x1001, 'chip': 'ESP32-C3', 'cpu_mhz': '160'}}
EVENT_NAMES = ['pre_led_on_1','pre_led_off_1','pre_led_on_2','pre_led_off_2',
               'pre_led_on_3','pre_led_off_3','active_begin','active_end',
               'post_led_on_1','post_led_off_1','post_led_on_2','post_led_off_2',
               'post_led_on_3','post_led_off_3']
IDLE_NAMES = ['idle_before_begin', 'idle_before_end', 'idle_after_begin', 'idle_after_end']
ACCEPTED = 'accepted_software_trace_alignment_descriptive_energy'

class GateError(RuntimeError):
    pass

def utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

def digest(path):
    h=hashlib.sha256(); size=0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            size += len(block); h.update(block)
    return {'bytes': size, 'sha256': h.hexdigest()}

def write_json(path, value):
    path=Path(path); temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def canonical_serial(value):
    return str(value or '').replace(':','').replace('-','').upper()

def check_board(ports, kind, requested=None):
    specification=BOARDS[kind]
    ports=list(ports)
    def matches(port):
        return (port.vid,port.pid)==(specification['vid'],specification['pid']) and canonical_serial(port.serial_number)==canonical_serial(specification['serial'])
    if requested:
        requested_ports=[p for p in ports if p.device.casefold()==requested.casefold()]
        if len(requested_ports)!=1 or not matches(requested_ports[0]):
            raise GateError('Порт '+requested+' не принадлежит требуемой прежней плате; загрузка запрещена.')
    matches_list=[p for p in ports if matches(p)]
    if len(matches_list)!=1:
        raise GateError('Нужна одна '+kind+' с SER='+specification['serial']+'. Чужие порты не открываются.')
    p=matches_list[0]
    return {'port':p.device,'vid':p.vid,'pid':p.pid,'serial_number':p.serial_number,'description':p.description}

def rediscover(comports,kind,timeout=45):
    deadline=time.monotonic()+timeout
    while True:
        try: return check_board(comports(),kind)
        except GateError:
            if time.monotonic()>=deadline: raise
            time.sleep(.25)

def plan_for(kind):
    plan=[{'index':0,'key':'pilot_coeff','phase':'pilot','repeat':0,'model':'coeff','board_kind':kind}]
    for repeat in range(1,6):
        rotation=(repeat-1)%5
        for model in MODELS[rotation:]+MODELS[:rotation]:
            plan.append({'index':len(plan),'key':f'block{repeat:02d}_{model}','phase':'campaign','repeat':repeat,'model':model,'board_kind':kind})
    return plan

def verify_sources(root):
    root=Path(root).resolve(); manifest=root/'KIT_MANIFEST.json'
    entries=read_json(manifest).get('files')
    if not isinstance(entries,list) or not entries: raise GateError('Пустой KIT_MANIFEST.json')
    seen=set()
    for entry in entries:
        name=entry.get('path',''); posix=PurePosixPath(name)
        path=(root/name).resolve()
        if not name or '\\' in name or posix.is_absolute() or '..' in posix.parts or name in seen or not path.is_relative_to(root) or not path.is_file():
            raise GateError('Неверный путь манифеста: '+name)
        if digest(path)!={'bytes':entry.get('bytes'),'sha256':entry.get('sha256')}:
            raise GateError('Файл пакета изменён: '+name)
        seen.add(name)
    required={'run_suite.py','bootstrap.py','PROTOCOL.json','project/platformio.ini','project/src/main.cpp','analyze_trace.py','cfn_decoder.py','FROZEN_COHORT_REFERENCE.json','summarize_results.py'}
    if read_json(root/'PROTOCOL.json').get('acquisition_protocol') == 'kanids-hw500-continuous-v1':
        required.update({'run_continuous.py','analyze_continuous.py','tests_continuous/test_analysis.py','tests_continuous/test_runner.py'})
    for folder in ('project/include','project/src','tests_runner'):
        required.update(p.relative_to(root).as_posix() for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    if not required.issubset(seen): raise GateError('Исходники отсутствуют в манифесте: '+str(required-seen))
    return entries

def copy_sources(root,destination,entries):
    for entry in entries+[{'path':'KIT_MANIFEST.json'}]:
        source=root/entry['path']; target=destination/entry['path']
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        if digest(source)!=digest(target): raise GateError('Копия исходника повреждена')

def make_archive(directory):
    """Manifest every payload, CRC-check the ZIP and atomically replace the snapshot."""
    directory=Path(directory)
    entries=[{'path':p.relative_to(directory).as_posix(),**digest(p)} for p in sorted(directory.rglob('*')) if p.is_file() and p != directory/'PACKET_MANIFEST.json' and '.tmp-' not in p.name]
    write_json(directory/'PACKET_MANIFEST.json',{'schema':'kanids-packet-v1','files':entries})
    archive=directory.with_suffix('.zip'); temp=archive.with_name(archive.name+'.tmp-'+uuid.uuid4().hex)
    with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for entry in entries+[{'path':'PACKET_MANIFEST.json'}]:
            zf.write(directory/entry['path'],directory.name+'/'+entry['path'])
    with zipfile.ZipFile(temp) as zf:
        bad=zf.testzip()
        if bad is not None: raise GateError('CRC ZIP: '+bad)
        for entry in entries:
            raw=zf.read(directory.name+'/'+entry['path'])
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
                raise GateError('ZIP изменился во время упаковки')
    temp.replace(archive)
    return archive

def verify_packet(directory):
    directory=Path(directory)
    entries=read_json(Path(directory)/'PACKET_MANIFEST.json')['files']
    actual={p.relative_to(directory).as_posix() for p in Path(directory).rglob('*') if p.is_file() and p != directory/'PACKET_MANIFEST.json'}
    if actual!={e['path'] for e in entries}: raise GateError('Состав сохранённого опыта изменён')
    for entry in entries:
        p=(Path(directory)/entry['path']).resolve()
        if not p.is_relative_to(Path(directory).resolve()) or digest(p)!={'bytes':entry['bytes'],'sha256':entry['sha256']}:
            raise GateError('Сохранённый опыт изменён: '+entry['path'])

def fields(line,tag):
    parts=line.split()
    if not parts or parts[0]!=tag: raise GateError('Ожидалось '+tag+': '+line)
    result={}
    for part in parts[1:]:
        if '=' not in part: raise GateError('Некорректная строка: '+line)
        key,value=part.split('=',1)
        if not key or not value or key in result: raise GateError('Повторное/пустое поле: '+line)
        result[key]=value
    return result

def integer(values,name):
    value=values.get(name,'')
    if not re.fullmatch(r'[0-9]+',str(value)): raise GateError('Нет целого поля '+name)
    return int(value)

def validate_info(lines,kind,model):
    result={}
    for line in lines:
        tag=line.split()[0]
        if tag in result: raise GateError('Повторный INFO: '+tag)
        result[tag]=fields(line,tag)
    if not {'HELLO','MODEL','COHORT','SYSTEM','INFO_DONE','RAW_IDS','WDT'}.issubset(result): raise GateError('Неполный INFO')
    if result['HELLO']!={'protocol':PROTOCOL}: raise GateError('Другой протокол прошивки')
    expected={'variant':model,'model_bytes':str(MODEL_BYTES[model]),'cohort_sha256':COHORT_SHA}
    if any(result['MODEL'].get(k)!=v for k,v in expected.items()): raise GateError('Другая модель/когорта')
    expected={'rows':'500','boundary':'flash_row_load_predict_checksum','model_placement':'flash'}
    if any(result['COHORT'].get(k)!=v for k,v in expected.items()): raise GateError('Другие строки/граница измерения')
    expected_ids=read_json(Path(__file__).resolve().parent/'FROZEN_COHORT_REFERENCE.json')['row_ids']
    if result['RAW_IDS']!={'values':','.join(map(str,expected_ids))}: raise GateError('Порядок 500 исходных строк отличается')
    common={'interrupts':'enabled','active_yield':'0','led_gpio':'13' if kind=='mega' else '8','led_active':'HIGH' if kind=='mega' else 'LOW','flash_bytes':'262144' if kind=='mega' else '4194304','timer':'micros' if kind=='mega' else 'esp_timer_get_time','timer_bits':'32' if kind=='mega' else '64'}
    if kind=='c3': common.update(wifi='not_initialized',bt='not_initialized')
    else: common['elapsed']='unsigned_modulo'
    if any(result['SYSTEM'].get(k)!=v for k,v in common.items()): raise GateError('SYSTEM: параметры измерения отличаются')
    if result['SYSTEM'].get('led_confirmed') not in ('0','1'): raise GateError('Некорректный LED state')
    if kind=='mega' and result['WDT']!={'policy':'unchanged_not_instrumented'}: raise GateError('Другая WDT policy')
    if kind=='c3' and (result['WDT'].get('policy')!='idle0_subscription_temporarily_suspended' or result['WDT'].get('idle_before') not in ('0','1','disabled') or result['WDT'].get('loop_before') not in ('0','disabled')): raise GateError('Неподходящее состояние WDT')
    for key in ('chip','cpu_mhz'):
        if result['SYSTEM'].get(key)!=BOARDS[kind][key]: raise GateError('SYSTEM: '+key)
    if kind=='c3' and canonical_serial(result['SYSTEM'].get('mac'))!=canonical_serial(BOARDS[kind]['serial']):
        raise GateError('MAC прошивки не совпадает с выбранной C3')
    if result['INFO_DONE'].get('state') not in ('idle','ready','done'): raise GateError('Прошивка не готова')
    if set(result)-{'HELLO','MODEL','COHORT','SYSTEM','INFO_DONE','WDT','RAW_IDS'}: raise GateError('Неизвестный INFO tag')
    return result

def validate_ready(lines, model):
    if len(lines)!=2: raise GateError('ARM должен вернуть CORRECTNESS и READY')
    correctness=fields(lines[0],'CORRECTNESS')
    if correctness.get('checked')!='500' or correctness.get('correct')!='1': raise GateError('500 предсказаний не прошли golden проверку')
    data=fields(lines[1],'READY')
    ready={k:integer(data,k) for k in ('count','target_us','cal_n','cal_us','checksum_per500','warmup_count')}
    if not (ready['target_us']==120000000 and ready['warmup_count']==500 and ready['checksum_per500']==EXPECTED_SUMS[model]
            and 500<=ready['cal_n']<=8192000 and ready['cal_n']%500==0 and (ready['cal_n']//500)&((ready['cal_n']//500)-1)==0 and 100000<=ready['cal_us']<=30000000
            and 500<=ready['count']<=2000000000 and ready['count']%500==0):
        raise GateError('READY вне протокола')
    expected=(ready['target_us']*ready['cal_n']//ready['cal_us']//500)*500
    if ready['count']!=expected: raise GateError('READY count не соответствует калибровке')
    return ready

def validate_done(lines,ready,kind=None):
    done=None; events=[]; watchdog=[]; started=False; terminal=False
    for line in lines:
        if terminal: raise GateError('Данные после WAIT_DONE')
        tag=line.split()[0]; obj=fields(line,tag)
        if tag=='PREP':
            if started or obj!={'sync':'begin'}: raise GateError('Неверный PREP')
            started=True
        elif tag=='DONE':
            if not started or done is not None: raise GateError('Неверный DONE')
            done={k:integer(obj,k) for k in ('count','active_us','checksum','expected','correct','timing_ok','wdt_restored')}
        elif tag=='EVENT':
            if done is None: raise GateError('EVENT до DONE')
            events.append({'name':obj.get('name'),'us':integer(obj,'us')})
        elif tag=='WDT':
            if obj.get('phase')=='before':
                if started or done is not None or watchdog: raise GateError('WDT before должен предшествовать PREP')
            elif obj.get('phase')=='after':
                if done is None or len(events)!=18 or len(watchdog)!=1: raise GateError('WDT after должен завершать EVENT')
            else: raise GateError('Неверная фаза WDT')
            watchdog.append(obj)
        elif tag=='WAIT_DONE':
            if obj!={'reset_required':'0','rearm_allowed':'1'}: raise GateError('Неверный WAIT_DONE')
            terminal=True
        else: raise GateError('Неожиданная строка: '+line)
    if done is None or not terminal: raise GateError('Опыт не завершён')
    expected=ready['count']//500*ready['checksum_per500']
    if not (done['count']==ready['count'] and done['checksum']==done['expected']==expected
            and done['correct']==done['timing_ok']==done['wdt_restored']==1 and 60000000<=done['active_us']<=180000000):
        raise GateError('Неверное число вызовов/checksum/длительность')
    names=[e['name'] for e in events]
    if len(names)!=18 or set(names)!=set(EVENT_NAMES+IDLE_NAMES): raise GateError('Неполный/повторный набор EVENT')
    if [n for n in names if n in EVENT_NAMES]!=EVENT_NAMES: raise GateError('Порядок маркеров неверный')
    times=[e['us'] for e in events]
    if times[0]>100000 or any(b<a for a,b in zip(times,times[1:])): raise GateError('Немонотонные EVENT')
    t={e['name']:e['us'] for e in events}; legacy=[t[n] for n in EVENT_NAMES]
    gaps=[2000000,2000000,4000000,2000000,2000000,10000000,done['active_us'],10000000,2000000,2000000,4000000,2000000,2000000]
    if any(abs((b-a)-gap)>100000 for a,b,gap in zip(legacy,legacy[1:],gaps)): raise GateError('Времена маркеров вне допуска 100 ms')
    for a,b in [('idle_before_end','active_begin'),('idle_after_begin','active_end')]:
        if t[a]!=t[b]: raise GateError('Границы idle/active не совпадают')
    if not (0<=t['idle_before_begin']-t['pre_led_off_3']<=5000 and 0<=t['post_led_on_1']-t['idle_after_end']<=5000): raise GateError('Границы idle/LED вне допуска')
    if kind=='mega' and watchdog: raise GateError('Неожиданные WDT события AVR')
    if kind=='c3':
        if len(watchdog)!=2 or [w.get('phase') for w in watchdog]!=['before','after']: raise GateError('Нет обеих квитанций WDT')
        before,after=watchdog
        if before.get('loop_before') not in ('0','disabled') or before.get('idle_before') not in ('0','1','disabled'): raise GateError('WDT before вне протокола')
        if before.get('suspended')!=('1' if before['idle_before']=='1' else '0') or after.get('restored')!='1' or any(before.get(k)!=after.get(k) for k in ('loop_before','idle_before','suspended')): raise GateError('WDT не восстановлен')
    if t['active_end']-t['active_begin']!=done['active_us']: raise GateError('EVENT и active_us различаются')
    return {'done':done,'events':events,'watchdog':watchdog,'software_protocol_pass':True,'ready':ready}

def run_command(command,directory,stem,record):
    env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['PYTHONUTF8']='1'
    info={'argv':command,'started_utc':utc(),'log':stem+'.log','child_python_io_encoding':'utf-8'}
    record.setdefault('commands',[]).append(info)
    print('\n> '+subprocess.list2cmdline(command),flush=True)
    with (Path(directory)/(stem+'.log')).open('w',encoding='utf-8',newline='\n') as stream:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
        try:
            for line in proc.stdout: stream.write(line); stream.flush(); print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            info['interrupted']=True; raise
        finally:
            proc.stdout.close(); info['finished_utc']=utc(); info['returncode']=proc.returncode
    if code: raise GateError('Команда завершилась с ошибкой: '+stem)

class SerialCapture:
    def __init__(self,device,directory):
        self.device=device; self.raw=(directory/'serial_raw.bin').open('xb')
        self.log=(directory/'serial.jsonl').open('x',encoding='utf-8',newline='\n')
        self.text=(directory/'serial.log').open('x',encoding='utf-8',newline='\n')
        self.buffer=bytearray(); self.pending=deque()
    def entry(self,direction,data):
        self.log.write(json.dumps({'utc':utc(),'host_monotonic_ns':time.perf_counter_ns(),'direction':direction,'hex':data.hex()})+'\n'); self.log.flush()
    def line(self,deadline):
        while time.monotonic()<deadline:
            if self.pending:
                line=self.pending.popleft().decode('ascii',errors='strict').rstrip('\r')
                if line: self.text.write(line+'\n'); self.text.flush(); print(line,flush=True); return line
                continue
            data=self.device.read(min(max(self.device.in_waiting,1),4096))
            if not data: continue
            self.raw.write(data); self.raw.flush(); self.entry('RX',data); self.buffer.extend(data)
            while b'\n' in self.buffer:
                value,_,rest=self.buffer.partition(b'\n'); self.pending.append(bytes(value)); self.buffer=bytearray(rest)
            if len(self.buffer)>8192: raise GateError('Слишком длинная строка USB')
        raise GateError('Время ожидания платы истекло')
    def send(self,command):
        data=(command+'\n').encode('ascii'); self.entry('TX',data)
        if self.device.write(data)!=len(data): raise GateError('Команда передана не полностью')
        self.device.flush()
    def until(self,terminal,timeout=60,boot_noise=False):
        result=[]; deadline=time.monotonic()+timeout; started=not boot_noise
        while True:
            line=self.line(deadline)
            if not started:
                if line.startswith('HELLO '): started=True
                elif line.startswith('ERROR '): raise GateError(line)
                else: continue
            result.append(line)
            if line.startswith('ERROR '): raise GateError(line)
            if line.startswith(terminal+' '): return result
    def close(self):
        for stream in (self.raw,self.log,self.text): stream.close()

def binary_names(kind):
    return ('firmware.elf','firmware.hex') if kind=='mega' else ('firmware.elf','firmware.bin','partitions.bin','bootloader.bin')

def env_name(kind,model): return kind+'_500_'+model

def copy_binaries(project,kind,model,destination):
    location=project/'.pio/build'/env_name(kind,model); destination.mkdir(parents=True,exist_ok=True); identities={}
    for name in binary_names(kind):
        source=location/name
        if not source.is_file(): raise GateError('Нет бинарника: '+str(source))
        identities[name]=digest(source); shutil.copy2(source,destination/name)
    if kind=='c3':
        from platformio.project.config import ProjectConfig
        from platformio.package.manager.tool import ToolPackageManager
        config=ProjectConfig.get_instance(str(project/'platformio.ini'))
        package=ToolPackageManager(config.get('platformio','packages_dir')).get_package('framework-arduinoespressif32@3.20017.241212')
        if package is None: raise GateError('Нет закреплённого framework-arduinoespressif32')
        source=Path(package.path)/'tools/partitions/boot_app0.bin'
        identities['boot_app0.bin']={**digest(source),'source_path':str(source),'flash_offset':'0xe000'}
        shutil.copy2(source,destination/'boot_app0.bin')
        shutil.copy2(Path(package.path)/'package.json',destination/'framework-package.json')
    audit_names=('firmware.map','firmware.nm.txt','firmware.disassembly.txt','firmware.audit.json')
    for name in audit_names:
        source=location/name
        if not source.is_file(): raise GateError('Нет обязательного аудита сборки: '+str(source))
        identities[name]=digest(source); shutil.copy2(source,destination/name)
    audit=read_json(location/'firmware.audit.json')
    if audit.get('environment')!=env_name(kind,model) or not all(audit.get(k) is True for k in ('active_symbol_present','prediction_symbol_present','batch_prediction_call_verified')): raise GateError('Аудит машинного кода не подтвердил вызов inference')
    if kind=='mega' and not (audit.get('near_progmem_checked') and 0<audit.get('near_progmem_max_end',0)<=65536 and 0<audit.get('static_sram_bytes',0)<4096): raise GateError('Аудит AVR memory placement не пройден')
    return identities

def assert_binaries(project,kind,model,identities):
    for name,identity in identities.items():
        source=Path(identity['source_path']) if name=='boot_app0.bin' else project/'.pio/build'/env_name(kind,model)/name
        if digest(source)!={k:identity[k] for k in ('bytes','sha256')}: raise GateError('Бинарник изменился: '+name)

def validate_cfn(path,active_us,used_hashes):
    from cfn_decoder import decode
    path=Path(path)
    if path.suffix.lower()!='.cfn' or not path.is_file(): raise GateError('Нужен существующий .cfn')
    identity=digest(path)
    if identity['sha256'] in used_hashes: raise GateError('Этот CFN уже использован в серии')
    metadata,rows=decode(path)
    if not metadata['decoded_to_exact_eof'] or metadata['sample_rate_header_sps']!=10: raise GateError('Нужен полный CFN при 10 sps')
    if not {0,1,4}.issubset({c['code'] for c in metadata['channels']}): raise GateError('Нужны VBUS, IBUS, PBUS')
    if any(abs(b[0]-a[0]-.1)>1e-7 for a,b in zip(rows,rows[1:])): raise GateError('Нарушена временная сетка CFN')
    span=rows[-1][0]-rows[0][0]
    if span<active_us/1e6+44: raise GateError('CFN не покрывает активную фазу и оба маркера')
    if metadata['start_current_mA']!=0 or metadata['stop_current_mA']!=0 or metadata['stop_time_s']!=5: raise GateError('Start/Stop CUR должны быть 0, Stop time=5 s')
    return {'file':identity,'metadata':metadata,'span_s':span,'structural_pass':True}

def collect_cfn(attempt,capture_path,record,used_hashes,prompt=input):
    from analyze_trace import analyze
    print('\nВ FNB58 нажмите Stop (квадрат), затем Сохранить.')
    print('Сохраните НОВЫЙ файл ровно сюда:\n'+str(capture_path))
    prompt('После сохранения нажмите Enter: ')
    if not capture_path.is_file(): raise GateError('CFN не найден: '+str(capture_path))
    destination=attempt/capture_path.name
    if destination.exists(): raise GateError('Запрещена перезапись CFN')
    before=digest(capture_path)
    shutil.copy2(capture_path,destination)
    if digest(destination)!=before or digest(capture_path)!=before: raise GateError('CFN изменился при копировании')
    # Preserve the supplied trace even when structural or scientific validation fails.
    record['cfn_supplied']={'path':str(capture_path),'saved_name':destination.name,**digest(destination)}
    write_json(attempt/'RUN_RECORD.json',record)
    started=datetime.fromisoformat(record['run_command_utc']).timestamp()
    if capture_path.stat().st_mtime < started-2: raise GateError('CFN сохранён до текущего RUN')
    audit=validate_cfn(destination,record['result']['done']['active_us'],used_hashes)
    analysis=analyze(destination,record)
    if analysis.get('status')!='ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_ESTIMATE': raise GateError('Анализ CFN не принят')
    write_json(attempt/'CFN_RECEIPT.json',audit); write_json(attempt/'TRACE_ANALYSIS.json',analysis)
    used_hashes.add(audit['file']['sha256'])
    return audit

def open_device(serial_module,board):
    device=serial_module.Serial(port=None,baudrate=115200,timeout=.25,write_timeout=3)
    device.dtr=True; device.rts=False; device.port=board['port']; device.open(); return device

def check_versions():
    if sys.version_info[:2]!=(3,11) or sys.maxsize<=2**32: raise GateError('Нужен Python 3.11 64-bit')
    versions={name:importlib.metadata.version(name) for name in ('platformio','pyserial','numpy','scipy')}
    pins={'platformio':'6.1.19','pyserial':'3.5','numpy':'2.3.5','scipy':'1.17.0'}
    if versions!=pins: raise GateError('Версии должны совпадать с requirements.txt; запускайте START_MEGA.cmd / START_C3.cmd.')
    return versions

def verify_resume(root,directory,kind):
    record=read_json(directory/'SUITE_RECORD.json')
    if record.get('schema')!='kanids-hw500-suite-v1' or record.get('board_kind')!=kind: raise GateError('Другая схема/плата сессии')
    if record.get('plan')!=plan_for(kind): raise GateError('План сессии изменён')
    if record.get('source_manifest')!=digest(root/'KIT_MANIFEST.json') or record.get('protocol_file')!=digest(root/'PROTOCOL.json'):
        raise GateError('Продолжение требует точно тот же пакет и протокол')
    if record.get('versions')!=check_versions(): raise GateError('Версии окружения изменились; продолжение запрещено')
    verify_sources(directory/'sources')
    accepted={}; used=set()
    listed={a['relative_path'] for a in record['attempts']}
    actual={p.relative_to(directory).as_posix() for p in (directory/'attempts').iterdir() if p.is_dir()} if (directory/'attempts').is_dir() else set()
    if actual!=listed: raise GateError('Есть незарегистрированная попытка после аварийного завершения; сохраните сессию для проверки')
    for attempt in record['attempts']:
        path=(directory/attempt['relative_path']).resolve()
        if not path.is_relative_to(directory.resolve()): raise GateError('Путь опыта вне сессии')
        verify_packet(path)
        if digest(path/'RUN_RECORD.json')!=attempt['record_file']: raise GateError('Квитанция опыта изменена')
        data=read_json(path/'RUN_RECORD.json')
        if data['plan_item']!=record['plan'][data['plan_item']['index']]: raise GateError('Опыт относится к другому плану')
        if data.get('cfn_supplied'): used.add(data['cfn_supplied']['sha256'])
        if data['status']==ACCEPTED:
            key=data['plan_item']['key']
            if key in accepted: raise GateError('Повторно принятый опыт')
            accepted[key]=data; used.add(data['cfn']['file']['sha256'])
    return record,accepted,used

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',required=True,choices=tuple(BOARDS)); parser.add_argument('--port')
    parser.add_argument('--bootstrap-evidence',type=Path); parser.add_argument('--resume',type=Path); parser.add_argument('--retry-incomplete',action='store_true')
    parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--build-only',action='store_true')
    args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({'board':BOARDS[args.board],'plan':plan_for(args.board),'hardware_access':False},ensure_ascii=False,indent=2)); return 0
    root=Path(__file__).resolve().parent; project=root/'project'; kind=args.board
    record=None; directory=None; current=None; attempt=None; device=capture=None; code=1
    try:
        entries=verify_sources(root); versions=check_versions()
        import serial
        from serial.tools import list_ports
        import numpy, scipy  # Verify analyzer dependencies before any upload.
        if args.resume:
            directory=args.resume.expanduser().resolve(); record,accepted,used=verify_resume(root,directory,kind)
            if not args.retry_incomplete and any(a['status']!=ACCEPTED for a in record['attempts']):
                raise GateError('Есть незавершённый опыт. Для новой отдельной попытки добавьте --retry-incomplete; прежние файлы сохраняются.')
            record.setdefault('resumptions',[]).append({'utc':utc(),'retry_incomplete':args.retry_incomplete})
        else:
            sid=datetime.now(timezone.utc).strftime(kind.upper()+'_HW500_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]
            directory=root/'runs'/sid; directory.mkdir(parents=True,exist_ok=False)
            record={'schema':'kanids-hw500-suite-v1','suite_id':sid,'board_kind':kind,'expected_board':BOARDS[kind],
                    'started_utc':utc(),'status':'incomplete','source_manifest':digest(root/'KIT_MANIFEST.json'),
                    'protocol_file':digest(root/'PROTOCOL.json'),'versions':versions,'python':sys.version,
                    'system':platform.platform(),'plan':plan_for(kind),'builds':{},'attempts':[],
                    'measurement_boundary':'flash_row_load_predict_checksum; whole-board USB supply',
                    'energy_status':'descriptive_USB_input_estimate_not_calibrated_metrology',
                    'meter':{'model':'FNIRSI FNB58','serial_observed':'104332','connection':'operator_confirmed_unchanged_from_previous_campaign','record_grid_sps':10,'start_current_mA':0,'stop_current_mA':0,'stop_time_s':5}}
            accepted={}; used=set(); copy_sources(root,directory/'sources',entries)
        if args.bootstrap_evidence:
            evidence=args.bootstrap_evidence.resolve(); verify_packet(evidence)
            destination=directory/'bootstrap'/evidence.name
            if destination.exists(): raise GateError('Квитанция bootstrap уже есть')
            shutil.copytree(evidence,destination)
            record.setdefault('bootstrap_receipts',[]).append(destination.relative_to(directory).as_posix())
        captures=root/'captures'/record['suite_id']; captures.mkdir(parents=True,exist_ok=True)
        if not args.build_only: record['board_observed']=check_board(list_ports.comports(),kind,args.port)
        write_json(directory/'SUITE_RECORD.json',record)
        base=[sys.executable,'-m','platformio']
        for model in MODELS:
            if model in record['builds']:
                assert_binaries(project,kind,model,record['builds'][model]['binaries']); continue
            destination=directory/'builds'/model
            if destination.exists(): raise GateError('Незавершённая сборка уже есть; начните новую сессию, сохранив прежнюю.')
            destination.mkdir(parents=True)
            build={'model':model,'environment':env_name(kind,model),'commands':[]}
            run_command(base+['run','-d',str(project),'-e',env_name(kind,model)],destination,'01_build',build)
            run_command(base+['pkg','list','-d',str(project),'-e',env_name(kind,model)],destination,'02_packages',build)
            verify_sources(root); build['binaries']=copy_binaries(project,kind,model,destination/'binaries')
            write_json(destination/'BUILD_RECORD.json',build); record['builds'][model]=build
            write_json(directory/'SUITE_RECORD.json',record)
        if args.build_only:
            record['status']='builds_pass_no_upload'; code=0; return code
        if 'meter_versions_operator' not in record:
            print('Схема питания по вашему подтверждению прежняя. Укажите версии FNB58/приложения, если видны; Enter = неизвестны.')
            record['meter_versions_operator']=input('Версии (необязательно): ').strip() or 'unknown_not_reported'
        record['status']='in_progress'; record.pop('error',None)
        led_confirmed=bool(record.get('led_observation',{}).get('confirmed'))
        for item in record['plan']:
            if item['key'] in accepted: continue
            if item['phase']=='campaign' and 'pilot_coeff' not in accepted: raise GateError('Отдельный пилот не принят')
            aid=item['key']+'_'+uuid.uuid4().hex[:8]; attempt=directory/'attempts'/aid; attempt.mkdir(parents=True,exist_ok=False)
            current={'schema':'kanids-hw500-run-v1','run_id':aid,'suite_id':record['suite_id'],'board_kind':kind,
                     'model':item['model'],'phase':item['phase'],'repeat':item['repeat'],'cohort_rows':500,
                     'plan_item':item,'started_utc':utc(),'status':'incomplete','commands':[],
                     'source_manifest':record['source_manifest'],'protocol_file':record['protocol_file'],
                     'binaries':record['builds'][item['model']]['binaries'],'build_path':'builds/'+item['model'],'sources_path':'sources'}
            write_json(attempt/'RUN_RECORD.json',current)
            board=check_board(list_ports.comports(),kind); current['board_before_upload']=board
            verify_sources(root); assert_binaries(project,kind,item['model'],current['binaries'])
            run_command(base+['run','-d',str(project),'-e',env_name(kind,item['model']),'-t','nobuild','-t','upload','--upload-port',board['port']],attempt,'03_upload',current)
            verify_sources(root); assert_binaries(project,kind,item['model'],current['binaries'])
            board=rediscover(list_ports.comports,kind); current['board_after_upload']=board
            device=open_device(serial,board); capture=SerialCapture(device,attempt)
            time.sleep(2); capture.send('INFO'); current['info']=validate_info(capture.until('INFO_DONE',60,boot_noise=True),kind,item['model'])
            print('\nПроверка LED: вспышки 2–4–2 s с паузами 2 s.')
            capture.send('LEDTEST')
            if capture.until('LEDTEST',30)!=['LEDTEST state=begin'] or capture.until('LEDTEST',30)!=['LEDTEST state=done']:
                raise GateError('LEDTEST не завершён')
            if not led_confirmed:
                answer=input('Видели три вспышки? Введите ДА: ').strip().casefold()
                if answer not in ('да','yes'): raise GateError('Физический LED не подтверждён')
                led_confirmed=True; record['led_observation']={'confirmed':True,'utc':utc(),'answer':answer,'attempt':aid}
                write_json(directory/'SUITE_RECORD.json',record)
            capture.send('CONFIRM_LED')
            if capture.until('LEDCONFIRM',10)!=['LEDCONFIRM confirmed=1']: raise GateError('Нет LEDCONFIRM')
            capture.send('ARM'); current['ready']=validate_ready(capture.until('READY',120),item['model'])
            current['stabilization']={'started_utc':utc(),'requested_seconds':30}
            print('Автоматическая выдержка 30 s после калибровки; пока не запускайте FNB58 запись.')
            quiet_start=time.monotonic(); time.sleep(30)
            current['stabilization'].update(finished_utc=utc(),host_elapsed_seconds=time.monotonic()-quiet_start)
            capture_path=captures/(kind+'_'+item['key']+'_'+aid.rsplit('_',1)[-1]+'.cfn')
            if capture_path.exists(): raise GateError('Имя CFN уже занято')
            current['requested_cfn_path']=str(capture_path); write_json(attempt/'RUN_RECORD.json',current)
            print('\n=== '+kind.upper()+' '+str(item['index']+1)+'/26: '+item['key']+' ===')
            print('FNB58: 10 sps; Start CUR=0; Stop CUR=0; VBUS/IBUS/PBUS включены.')
            print('Новая запись (лист) → ▶. Убедитесь, что TIME растёт.\nПосле опыта сохранить: '+str(capture_path))
            answer=input('Когда TIME растёт, нажмите Enter (или Q для остановки): ').strip().casefold()
            if answer=='q': raise KeyboardInterrupt
            current['meter_TIME_increasing_operator_confirmed']=True; current['run_command_utc']=utc(); write_json(attempt/'RUN_RECORD.json',current); capture.send('RUN')
            print('Ждите около 3 минут. Не меняйте питание, Reset, кабель и настройки.')
            current['result']=validate_done(capture.until('WAIT_DONE',360),current['ready'],kind)
            capture.close(); capture=None; device.close(); device=None
            current['status']='software_pass_trace_pending'; write_json(attempt/'RUN_RECORD.json',current)
            current['cfn']=collect_cfn(attempt,capture_path,current,used)
            current['status']=ACCEPTED; current['finished_utc']=utc(); write_json(attempt/'RUN_RECORD.json',current)
            make_archive(attempt)
            record['attempts'].append({'relative_path':attempt.relative_to(directory).as_posix(),'status':current['status'],'record_file':digest(attempt/'RUN_RECORD.json')})
            accepted[item['key']]=current; current=None; attempt=None
            write_json(directory/'SUITE_RECORD.json',record); archive=make_archive(directory)
            print('Опыт принят и сохранён. Завершено '+str(len(accepted))+'/26. ZIP: '+str(archive))
            if item['phase']=='pilot': print('Отдельный пилот прошёл проверку 500 ответов, CFN и маркеров. Далее 25 опытов по фиксированному плану.')
        record['status']='all_26_runs_accepted_descriptive_energy'
        write_json(directory/'SUITE_RECORD.json',record)
        from summarize_results import summarize_session
        record['summary_status']=summarize_session(directory)['status']; code=0
    except KeyboardInterrupt:
        if record is not None: record['error']='interrupted_by_operator'
        print('\nОстановлено. Плата может ещё выполнять серию. Сохраните текущую запись FNB58; прежние результаты сохранены.',file=sys.stderr)
    except Exception as exc:
        if record is not None: record['error']=type(exc).__name__+': '+str(exc)
        print('\nОСТАНОВКА: '+str(exc)+'\nСледующий опыт не запускается.',file=sys.stderr)
    finally:
        if capture: capture.close()
        if device: device.close()
        if record is not None and directory is not None:
            if current is not None and attempt is not None:
                current['error']=record.get('error','incomplete'); current['finished_utc']=utc()
                write_json(attempt/'RUN_RECORD.json',current); make_archive(attempt)
                record['attempts'].append({'relative_path':attempt.relative_to(directory).as_posix(),'status':current['status'],'record_file':digest(attempt/'RUN_RECORD.json')})
            record['updated_utc']=utc(); record['accepted_count']=len(accepted)
            write_json(directory/'SUITE_RECORD.json',record); archive=make_archive(directory)
            print('\nSEND_THIS_ZIP: '+str(archive))
            if code: print('Для продолжения: '+subprocess.list2cmdline([sys.executable,str(root/'run_suite.py'),'--board',kind,'--resume',str(directory),'--retry-incomplete']))
    return code

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(errors='replace'); sys.stderr.reconfigure(errors='replace')
    raise SystemExit(main())
