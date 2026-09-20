#!/usr/bin/env python3
"""Identity-locked RAM diagnostics on five frozen models and three fresh uploads."""
from __future__ import annotations
import argparse
from collections import deque
import csv
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
import threading
import time
import uuid
import zipfile
sys.dont_write_bytecode = True
MODELS = ('coeff', 'lut', 'mlp', 'kanml', 'dt5')
MODEL_BYTES = dict(coeff=254, lut=20554, mlp=760, kanml=5244, dt5=285)
EXPECTED_SUMS = dict(coeff=256, lut=256, mlp=254, kanml=250, dt5=249)
PROTOCOL = 'kanids-ram500-v1'
C3_PROTOCOL = 'kanids-ram500-v2'
COHORT_SHA = '20c53b571bfaef91f5fd26b19e5960e3fa85ed48b38c78c8527245b921ebe86f'
PINS = {'platformio': '6.1.19', 'pyserial': '3.5'}
BOARDS = {'mega': {'serial': '14532303532351804271', 'vid': 0x2341, 'pid': 0x0042, 'chip': 'ATmega2560', 'cpu_mhz': '16'},
          'c3': {'serial': '10:00:3B:CB:8D:70', 'vid': 0x303A, 'pid': 0x1001, 'chip': 'ESP32-C3', 'cpu_mhz': '160'}}
ACCEPTED = 'accepted_observed_ram_components'
class GateError(RuntimeError):
    pass

class TransportTimeout(GateError):
    """Only an absent/incomplete response permits an idempotent cached reply request."""
    pass

def protocol_for(kind):
    return C3_PROTOCOL if kind=='c3' else PROTOCOL

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


class SerialCapture:
    def __init__(self,device,directory):
        self.directory=directory
        self.device=device; self.raw=(directory/'serial_raw.bin').open('xb')
        self.log=(directory/'serial.jsonl').open('x',encoding='utf-8',newline='\n')
        self.text=(directory/'serial.log').open('x',encoding='utf-8',newline='\n')
        self.buffer=bytearray(); self.pending=deque()
        self.events=[]; self.received_lines=[]; self.closed=False
    def snapshot(self):
        return {'pending_partial_hex':bytes(self.buffer).hex(),
                'pending_partial_ascii':bytes(self.buffer).decode('ascii',errors='backslashreplace'),
                'pending_complete_lines_hex':[value.hex() for value in self.pending],
                'last_received_lines':self.received_lines[-16:], 'recovery_events':self.events}
    def trace(self,kind,**details):
        self.events.append({'utc':utc(),'event':kind,**details})
        write_json(self.directory/'TRANSPORT_STATE.json',self.snapshot())
    def entry(self,direction,data):
        self.log.write(json.dumps({'utc':utc(),'host_monotonic_ns':time.perf_counter_ns(),'direction':direction,'hex':data.hex()})+'\n'); self.log.flush()
    def line(self,deadline):
        while time.monotonic()<deadline:
            if self.pending:
                line=self.pending.popleft().decode('ascii',errors='strict').rstrip('\r')
                if line:
                    self.received_lines.append(line)
                    self.text.write(line+'\n'); self.text.flush(); print(line,flush=True); return line
                continue
            data=self.device.read(min(max(self.device.in_waiting,1),4096))
            if not data: continue
            self.raw.write(data); self.raw.flush(); self.entry('RX',data); self.buffer.extend(data)
            while b'\n' in self.buffer:
                value,_,rest=self.buffer.partition(b'\n'); self.pending.append(bytes(value)); self.buffer=bytearray(rest)
            if len(self.buffer)>8192: raise GateError('Слишком длинная строка USB')
        raise TransportTimeout('Время ожидания платы истекло; незавершённая строка='+repr(bytes(self.buffer).decode('ascii',errors='backslashreplace')))
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
        if self.closed:return
        write_json(self.directory/'TRANSPORT_STATE.json',self.snapshot())
        if self.buffer or self.pending:
            self.text.write('[CAPTURE END: pending RX evidence; not a firmware line] '+json.dumps(self.snapshot(),ensure_ascii=False)+'\n')
            self.text.flush()
        for stream in (self.raw,self.log,self.text): stream.close()
        self.closed=True

    def c3_frame(self,marker,run_token,retry_command,timeout=30,max_retries=2,baseline_lines=None):
        """Read a token-bound frame; retries only request already frozen board state.

        Port, RX buffer and unfinished frame remain intact. A leading newline in
        the cached reply closes a stale partial line; only a new marker after an
        explicitly logged timeout may replace that incomplete frame. A late,
        complete baseline reply before CHECK is accepted only when it is exactly
        equal to the already validated baseline, and is logged as a duplicate.
        """
        if not re.fullmatch(r'[0-9a-f]{16}',run_token):raise GateError('Invalid planned run token')
        if marker not in ('REPORT_BEGIN','CONTROL_BEGIN'):raise GateError('Invalid frame marker')
        active=None; payload=[]; retries=0; replacement_allowed=False; run_begin_seen=False
        frozen_prefix={'REPORT_BEGIN':list(baseline_lines or []),'CONTROL_BEGIN':[]}
        preframe_partial=None; deadline=time.monotonic()+timeout
        while True:
            try:
                value=self.line(deadline)
            except TransportTimeout as exc:
                self.trace('frame_timeout',marker=marker,run_token=run_token,retry_number=retries,
                           active_marker=active,incomplete_frame_lines=list(payload),
                           pending_partial_hex=bytes(self.buffer).hex(),error=str(exc))
                if retries>=max_retries:
                    raise TransportTimeout('C3 '+marker+' incomplete after '+str(retries)+' cached reply retries; '+str(exc)) from exc
                if self.buffer:
                    fragment=bytes(self.buffer).decode('ascii',errors='strict').rstrip('\r')
                    if active is None:
                        candidates=[marker+' run_token='+run_token]
                        if marker=='CONTROL_BEGIN' and baseline_lines is not None:candidates.append('REPORT_BEGIN run_token='+run_token)
                        if marker=='REPORT_BEGIN' and not run_begin_seen:candidates.append('RUN_BEGIN run_token='+run_token)
                        known=any(candidate.startswith(fragment) for candidate in candidates)
                    else:
                        expected=('BEGIN','RAM_RESULT','DONE') if active=='REPORT_BEGIN' else ('CONTROL',)
                        tag_expected=expected[len(payload)]
                        known=tag_expected.startswith(fragment) or fragment.startswith(tag_expected+' ')
                    if known:
                        partial_tokens=re.findall(r'(?:^|\s)run_token=([^\s]+)',fragment)
                        if len(partial_tokens)>1 or any(not run_token.startswith(token) for token in partial_tokens):
                            raise GateError('Wrong or repeated run token in partial response: '+fragment)
                        preframe_partial=fragment
                replacement_allowed=True; retries+=1
                self.send(retry_command+' '+run_token)
                self.trace('cached_reply_requested',command=retry_command,run_token=run_token,retry_number=retries)
                deadline=time.monotonic()+timeout
                continue
            tag=value.split()[0]
            if any(marker_text in value for marker_text in ('HELLO protocol=','ESP-ROM:','ERROR reason=','Guru Meditation')):
                raise GateError('C3 error/reset within response while awaiting '+marker+': '+value)
            if replacement_allowed and preframe_partial is not None and value==preframe_partial:
                self.trace('known_partial_line_discarded',marker=active,run_token=run_token,line=value)
                preframe_partial=None;continue
            if tag in ('ERROR','HELLO') or value.startswith(('ESP-ROM:','rst:','Guru Meditation')):
                raise GateError('C3 error/reset while awaiting '+marker+': '+value)
            # Never accept a complete line carrying a different run identity.
            tokens=re.findall(r'(?:^|\s)run_token=([^\s]+)',value)
            if tokens and tokens!=[run_token]:raise GateError('Wrong or repeated run token: '+value)
            if tag in ('REPORT_BEGIN','CONTROL_BEGIN'):
                if fields(value,tag)!={'run_token':run_token}:raise GateError('Wrong frame header: '+value)
                if tag!=marker and not (marker=='CONTROL_BEGIN' and tag=='REPORT_BEGIN' and baseline_lines is not None):
                    raise GateError('Unexpected frame marker: '+value)
                if active is not None:
                    if not replacement_allowed:raise GateError('Unexpected replacement of an incomplete frame')
                    self.trace('incomplete_frame_discarded',marker=active,run_token=run_token,lines=list(payload))
                active=tag;payload=[];replacement_allowed=False;preframe_partial=None
                continue
            if active is None:
                if tag=='RUN_BEGIN' and marker=='REPORT_BEGIN' and not run_begin_seen:
                    if fields(value,tag)!={'run_token':run_token}:raise GateError('Wrong RUN_BEGIN: '+value)
                    run_begin_seen=True;self.trace('run_acknowledged',run_token=run_token);continue
                raise GateError('Unframed C3 response: '+value)
            expected=('BEGIN','RAM_RESULT','DONE') if active=='REPORT_BEGIN' else ('CONTROL',)
            if tag!=expected[len(payload)]:raise GateError('Unexpected C3 frame line: '+value)
            parsed=fields(value,tag)
            if tag in ('RAM_RESULT','DONE','CONTROL') and parsed.get('run_token')!=run_token:
                raise GateError('Missing/wrong run token in complete frame line: '+value)
            prefix=frozen_prefix[active];index=len(payload)
            if index<len(prefix):
                if value!=prefix[index]:raise GateError('Cached reply changed a previously complete frame line: '+value)
            else:prefix.append(value)
            payload.append(value)
            if len(payload)==len(expected):
                if active==marker:return payload
                if payload!=baseline_lines:raise GateError('Cached baseline reply differs from saved baseline')
                self.trace('identical_cached_baseline_discarded',run_token=run_token,lines=list(payload))
                active=None;payload=[];replacement_allowed=False

    def c3_verify_tail(self,run_token,baseline_lines,control_lines,quiet_timeout=.75,max_duration=3):
        """Validate queued cached replies before accepting/closing this boot."""
        finish=time.monotonic()+max_duration; quiet_deadline=time.monotonic()+quiet_timeout
        deadline=min(finish,quiet_deadline)
        active=None;payload=[]
        while True:
            try:value=self.line(deadline)
            except TransportTimeout as exc:
                if quiet_deadline>finish:raise GateError('C3 trailing replies did not settle within bounded drain') from exc
                if active is not None or self.buffer or self.pending:
                    raise GateError('Incomplete C3 trailing reply after positive control; '+str(exc)) from exc
                self.trace('terminal_quiet_verified',run_token=run_token,quiet_timeout_s=quiet_timeout)
                return
            if time.monotonic()>=finish:raise GateError('C3 trailing replies did not settle within bounded drain')
            quiet_deadline=time.monotonic()+quiet_timeout;deadline=min(finish,quiet_deadline)
            tag=value.split()[0]
            if active is None:
                if tag not in ('REPORT_BEGIN','CONTROL_BEGIN') or fields(value,tag)!={'run_token':run_token}:
                    raise GateError('Unexpected C3 trailing frame: '+value)
                active=tag;payload=[];continue
            expected=baseline_lines if active=='REPORT_BEGIN' else control_lines
            if value!=expected[len(payload)]:raise GateError('C3 trailing cached reply differs from saved frame: '+value)
            payload.append(value)
            if len(payload)==len(expected):
                self.trace('identical_terminal_cached_frame_discarded',marker=active,run_token=run_token,lines=list(payload))
                active=None;payload=[]


def open_device(serial_module,board):
    device=serial_module.Serial(port=None,baudrate=115200,timeout=.25,write_timeout=3)
    device.dtr=True; device.rts=False; device.port=board['port']; device.open(); return device

def plan_for(kind):
    result=[]
    for repeat in range(1, 4):
        rotation=repeat-1
        for model in MODELS[rotation:]+MODELS[:rotation]:
            result.append({'index': len(result), 'key': f'repeat{repeat:02d}_{model}',
                           'repeat': repeat, 'model': model, 'board_kind': kind})
    return result

def verify_sources(root):
    root=Path(root).resolve()
    entries=read_json(root/'KIT_MANIFEST.json').get('files')
    if not isinstance(entries,list) or not entries: raise GateError('Empty KIT_MANIFEST.json')
    seen=set()
    for entry in entries:
        name=entry.get('path',''); posix=PurePosixPath(name); path=(root/name).resolve()
        if not name or '\\' in name or posix.is_absolute() or '..' in posix.parts or name in seen or not path.is_relative_to(root) or not path.is_file():
            raise GateError('Invalid manifest path: '+name)
        if digest(path)!={'bytes':entry.get('bytes'),'sha256':entry.get('sha256')}:
            raise GateError('Файл пакета изменён: '+name)
        seen.add(name)
    required={'run_ram.py','bootstrap.py','requirements.txt','START_MEGA.cmd','START_C3.cmd','PROTOCOL.json',
              'project/platformio.ini','project/src/main.cpp','project/save_build_artifacts.py',
              'project/FROZEN_HEADERS.json','FROZEN_COHORT_REFERENCE.json'}
    for folder in ('project/include','project/src','tests_runner'):
        required.update(p.relative_to(root).as_posix() for p in (root/folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts)
    if not required.issubset(seen): raise GateError('Исходники отсутствуют в манифесте: '+str(sorted(required-seen)))
    frozen=read_json(root/'project/FROZEN_HEADERS.json')
    for entry in frozen.get('files',[]):
        path=(root/'project'/entry['path']).resolve()
        if not path.is_relative_to(root/'project') or digest(path)!={k:entry[k] for k in ('bytes','sha256')}:
            raise GateError('Замороженный заголовок изменён: '+entry['path'])
    if not frozen.get('files'): raise GateError('Нет frozen headers')
    return entries

def check_versions():
    if sys.version_info[:2]!=(3,11) or sys.maxsize<=2**32: raise GateError('Нужен Python 3.11 64-bit')
    versions={name:importlib.metadata.version(name) for name in PINS}
    if versions!=PINS: raise GateError('Версии должны совпадать с requirements.txt; используйте START_MEGA.cmd / START_C3.cmd')
    return versions

def run_command(command,directory,stem,record,timeout=1800):
    env=os.environ.copy(); env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    info={'argv':command,'started_utc':utc(),'log':stem+'.log','timeout_s':timeout,'encoding':'utf-8'}
    record.setdefault('commands',[]).append(info)
    print('\n> '+subprocess.list2cmdline(command),flush=True)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    # A reader thread streams merged stdout/stderr while the main thread enforces a wall deadline.
    with (directory/(stem+'.log')).open('w',encoding='utf-8',newline='\n') as stream:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                              encoding='utf-8',errors='replace',env=env,cwd=Path(__file__).resolve().parent)
        read_errors=[]
        def consume():
            try:
                for line in proc.stdout:
                    stream.write(line); stream.flush(); print(line,end='',flush=True)
            except BaseException as exc: read_errors.append(exc)
        worker=threading.Thread(target=consume,daemon=True); worker.start()
        try:
            code=proc.wait(timeout=timeout)
        except BaseException:
            info['interrupted_or_timeout']=True
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,check=False)
            else:
                proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
            raise
        finally:
            worker.join(timeout=10)
            info.update(finished_utc=utc(),returncode=proc.returncode)
        if worker.is_alive(): raise GateError('Дочерняя команда оставила открытый поток вывода')
        proc.stdout.close()
        if read_errors: raise GateError('Ошибка чтения вывода команды: '+repr(read_errors[0]))
    if code: raise GateError('Команда завершилась с ошибкой: '+stem)
    return info

def validate_info(lines,kind,model,root=None):
    result={}
    for line in lines:
        tag=line.split()[0]
        if tag in result: raise GateError('Repeated INFO tag: '+tag)
        result[tag]=fields(line,tag)
    required={'HELLO','MODEL','COHORT','RAW_IDS','SYSTEM','INFO_DONE'}
    if set(result)!=required: raise GateError('Incomplete/unexpected INFO tags')
    if result['HELLO']!={'protocol':protocol_for(kind)}: raise GateError('Wrong firmware protocol')
    for key,value in {'variant':model,'model_bytes':str(MODEL_BYTES[model]),'cohort_sha256':COHORT_SHA}.items():
        if result['MODEL'].get(key)!=value: raise GateError('Wrong model/cohort: '+key)
    for key,value in {'rows':'500','order':'attack_normal_interleaved','boundary':'flash_row_load_predict_checksum','model_placement':'flash'}.items():
        if result['COHORT'].get(key)!=value: raise GateError('Wrong cohort boundary: '+key)
    expected_ids=read_json((Path(root) if root else Path(__file__).resolve().parent)/'FROZEN_COHORT_REFERENCE.json')['row_ids']
    if result['RAW_IDS']!={'values':','.join(map(str,expected_ids))}: raise GateError('Wrong raw row ordering')
    for key in ('chip','cpu_mhz'):
        if result['SYSTEM'].get(key)!=BOARDS[kind][key]: raise GateError('Wrong SYSTEM '+key)
    if kind=='c3':
        if canonical_serial(result['SYSTEM'].get('mac'))!=canonical_serial(BOARDS[kind]['serial']): raise GateError('Wrong firmware MAC')
        if not re.fullmatch(r'v4\.4\.7(?:[-+_][^\s]+)?',result['SYSTEM'].get('idf','')): raise GateError('Expected pinned ESP-IDF 4.4.7')
        for key,value in {'heap_domain':'internal_8bit','watermark_units':'bytes','wifi':'not_initialized','bt':'not_initialized','wdt':'unchanged'}.items():
            if result['SYSTEM'].get(key)!=value: raise GateError('Wrong C3 runtime contract: '+key)
        if integer(result['SYSTEM'],'loop_stack_reserved_bytes')!=8192: raise GateError('Wrong Arduino loop stack size')
    elif any(result['SYSTEM'].get(k)!=v for k,v in {'sram_capacity_bytes':'8192','interrupts':'enabled','wdt':'unchanged'}.items()):
        raise GateError('Wrong Mega runtime contract')
    if result['INFO_DONE']!={'state':'idle'}: raise GateError('A fresh upload/boot is required for every repeat')
    return result

# Board-specific metric validation is defined below after the firmware contract is frozen.
def validate_result(lines,kind,model,build_audit=None,run_token=None):
    if len(lines)!=3: raise GateError('RUN must return BEGIN, RAM_RESULT and DONE')
    begin=fields(lines[0],'BEGIN')
    if begin!={'rows':'500','passes':'3','scope':'load_predict_reference_check'}: raise GateError('Wrong RAM measurement scope')
    data=fields(lines[1],'RAM_RESULT'); done=fields(lines[2],'DONE')
    expected={'board':kind,'model':model,'rows':'500','passes':'3','checked':'1500','mismatches':'0',
              'checksum':str(3*EXPECTED_SUMS[model]),'checksum_per500':str(EXPECTED_SUMS[model]),'correct':'1'}
    if any(data.get(k)!=v for k,v in expected.items()): raise GateError('RAM run correctness/counter gate failed')
    expected_done={'correct':'1','reset_required':'1'}
    if run_token is not None:
        if kind!='c3' or not re.fullmatch(r'[0-9a-f]{16}',run_token):raise GateError('Invalid expected run token')
        if data.get('run_token')!=run_token:raise GateError('RAM_RESULT run token differs from planned run')
        expected_done['run_token']=run_token
    if done!=expected_done: raise GateError('RAM DONE gate failed')
    validate_metrics(data,kind,build_audit)
    result={'begin':begin,'ram':data,'done':done,'software_protocol_pass':True}
    if run_token is not None:result['run_token']=run_token
    return result

def validate_control(lines,kind,result):
    if len(lines)!=1: raise GateError('CONTROL must return exactly one line')
    data=fields(lines[0],'CONTROL')
    if kind=='c3' and result.get('run_token') is not None and data.get('run_token')!=result['run_token']:
        raise GateError('CONTROL run token differs from saved baseline')
    detail='board='+kind+' '+ ' '.join(key+'='+data.get(key,'missing') for key in (
        'stack_pass','heap_pass','correct','stack_before_bytes','stack_after_bytes',
        'heap_before','heap_during','heap_after'))
    if kind=='mega':
        detail+=' '+' '.join(key+'='+data.get(key,'missing') for key in (
            'heap_alloc_ok','heap_malloc_calls','heap_free_calls'))
    if any(data.get(key)!='1' for key in ('stack_pass','heap_pass','correct')):
        raise GateError('Instrumentation positive control failed: '+detail)
    try:
        validate_control_metrics(data,kind,result)
    except GateError as exc:
        raise GateError(str(exc)+'; '+detail) from exc
    return data

def env_name(kind,model): return kind+'_ram_'+model

def copy_binaries(project,kind,model,destination):
    location=project/'.pio/build'/env_name(kind,model); destination.mkdir(parents=True,exist_ok=False)
    names=['firmware.elf','firmware.map','firmware.nm.txt','firmware.disassembly.txt','firmware.audit.json',
           'firmware.sections.txt','firmware.stack_usage.json']
    names+=['firmware.hex'] if kind=='mega' else ['firmware.bin','partitions.bin','bootloader.bin']
    identities={}
    for name in names:
        source=location/name
        if not source.is_file(): raise GateError('Нет обязательного файла сборки: '+str(source))
        shutil.copy2(source,destination/name); identities[name]=digest(source)
    su_files=sorted(location.rglob('*.su'))
    if not su_files: raise GateError('Нет .su файлов stack-usage от компилятора')
    for source in su_files:
        relative=source.relative_to(location).as_posix(); target=destination/'stack_usage'/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        identities['stack_usage/'+relative]=digest(source)
    audit=read_json(location/'firmware.audit.json')
    validate_build_audit(audit,kind,model)
    if kind=='c3':
        from platformio.project.config import ProjectConfig
        from platformio.package.manager.tool import ToolPackageManager
        config=ProjectConfig.get_instance(str(project/'platformio.ini'))
        package=ToolPackageManager(config.get('platformio','packages_dir')).get_package('framework-arduinoespressif32@3.20017.241212')
        if package is None: raise GateError('Нет закреплённого framework-arduinoespressif32')
        source=Path(package.path)/'tools/partitions/boot_app0.bin'
        shutil.copy2(source,destination/'boot_app0.bin')
        identities['boot_app0.bin']={**digest(source),'source_path':str(source),'flash_offset':'0xe000'}
        shutil.copy2(Path(package.path)/'package.json',destination/'framework-package.json')
    return identities,audit

def assert_binaries(project,kind,model,identities):
    location=project/'.pio/build'/env_name(kind,model)
    for name,identity in identities.items():
        if name=='boot_app0.bin': source=Path(identity['source_path'])
        else: source=location/(name[len('stack_usage/'):] if name.startswith('stack_usage/') else name)
        if digest(source)!={k:identity[k] for k in ('bytes','sha256')}: raise GateError('Бинарник/аудит изменился: '+name)

def connect_info(serial_module,comports,kind,model,attempt,root,timeout=60):
    deadline=time.monotonic()+timeout; failures=[]
    while time.monotonic()<deadline:
        device=capture=None
        try:
            board=check_board(comports(),kind)
            device=open_device(serial_module,board)
            capture_dir=attempt/('serial_connection_%02d'%len(failures)); capture_dir.mkdir()
            capture=SerialCapture(device,capture_dir)
            time.sleep(2)
            capture.send('INFO')
            lines=capture.until('INFO_DONE',min(8,max(1,deadline-time.monotonic())),boot_noise=True)
            info=validate_info(lines,kind,model,root)
            return device,capture,board,info,failures
        except (OSError,serial_module.SerialException) as exc:
            failures.append({'utc':utc(),'error':repr(exc)})
        except GateError as exc:
            # Retry only transport absence/timeouts, never a semantically wrong device response.
            message=str(exc)
            if message.startswith('Время ожидания') or message.startswith('Нужна одна '):
                failures.append({'utc':utc(),'error':message})
            else:
                if capture: capture.close()
                if device: device.close()
                raise
        if capture: capture.close()
        if device: device.close()
        time.sleep(.5)
    write_json(attempt/'CONNECTION_FAILURES.json',failures)
    raise GateError('Плата не вернулась после загрузки за '+str(timeout)+' s')

def acquire_one(root,project,directory,suite,item,serial_module,comports):
    kind=suite['board_kind']; model=item['model']; aid=item['key']
    attempt=directory/'attempts'/aid; attempt.mkdir(parents=True,exist_ok=False)
    current={'schema':'kanids-ram500-run-v1','plan_item':item,'board_kind':kind,'model':model,
             'repeat':item['repeat'],'started_utc':utc(),'status':'incomplete','commands':[],
             'build_path':'builds/'+model,'fresh_boot_method':'upload_each_repeat',
             'source_manifest':suite['source_manifest'],'build_identities':suite['builds'][model]['binaries']}
    current['build_ram']={key:suite['builds'][model]['audit'][key] for key in ('static_sram_bytes','diagnostic_globals_bytes')}
    if kind=='c3': current['build_ram']['iram_code_bytes']=suite['builds'][model]['audit']['iram_code_bytes']
    device=capture=None
    try:
        board=check_board(comports(),kind); current['board_before_upload']=board
        verify_sources(root); assert_binaries(project,kind,model,current['build_identities'])
        run_command([sys.executable,'-m','platformio','run','-d',str(project),'-e',env_name(kind,model),
                     '-t','nobuild','-t','upload','--upload-port',board['port']],attempt,'03_upload',current,timeout=180)
        verify_sources(root); assert_binaries(project,kind,model,current['build_identities'])
        device,capture,board,info,failures=connect_info(serial_module,comports,kind,model,attempt,root)
        current.update(board_after_upload=board,info=info,connection_retries=failures)
        current['baseline_start_utc']=utc()
        if kind=='c3':
            current['run_token']=uuid.uuid4().hex[:16]
            capture.send('RUN '+current['run_token'])
            baseline_lines=capture.c3_frame('REPORT_BEGIN',current['run_token'],'RESULT')
            current['result']=validate_result(baseline_lines,kind,model,suite['builds'][model]['audit'],current['run_token'])
        else:
            capture.send('RUN')
            baseline_lines=capture.until('DONE',180)
            current['result']=validate_result(baseline_lines,kind,model,suite['builds'][model]['audit'])
        current['baseline_serial_lines']=baseline_lines
        current['baseline_end_utc']=utc()
        write_json(attempt/'BASELINE_RECORD.json',current)
        # Positive controls deliberately run after all baseline values are already captured.
        current['control_start_utc']=utc()
        if kind=='c3':
            capture.send('CHECK '+current['run_token'])
            control_lines=capture.c3_frame('CONTROL_BEGIN',current['run_token'],'CHECK',baseline_lines=baseline_lines)
        else:
            capture.send('CONTROL');control_lines=capture.until('CONTROL',60)
        current['control']=validate_control(control_lines,kind,current['result'])
        if kind=='c3':capture.c3_verify_tail(current['run_token'],baseline_lines,control_lines)
        current['control_end_utc']=utc(); current['status']=ACCEPTED
        return current
    except BaseException as exc:
        current['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        if capture:
            if hasattr(capture,'snapshot'):current['transport_final_state']=capture.snapshot()
            capture.close()
        if device: device.close()
        current['finished_utc']=utc(); write_json(attempt/'RUN_RECORD.json',current)
        suite['attempts'].append({'key':aid,'status':current['status'],'path':attempt.relative_to(directory).as_posix(),
                                  'record':digest(attempt/'RUN_RECORD.json')})
        write_json(directory/'SUITE_RECORD.json',suite)

def run_plan(root,project,directory,suite,serial_module,comports,acquire=acquire_one):
    records=[]
    for item in suite['plan']:
        print('\nRAM: %d/15 — %s, повтор %d/3'%(item['index']+1,item['model'],item['repeat']),flush=True)
        result=acquire(root,project,directory,suite,item,serial_module,comports)
        if result.get('status')!=ACCEPTED: raise GateError('Incomplete RAM run; later models are not started')
        records.append(result)
    return records

def make_summary(records,kind):
    expected=plan_for(kind)
    if len(records)!=15 or [r.get('plan_item') for r in records]!=expected or any(r.get('status')!=ACCEPTED for r in records):
        raise GateError('A complete summary requires all 15 accepted fresh-boot runs')
    output=[]
    for model in MODELS:
        group=[r for r in records if r['model']==model]
        row={'board':kind,'model':model,'fresh_boot_repeats':3,'rows_per_pass':500,'passes_per_boot':3,
             'scope':'observed_components_instrumented_firmware_not_whole_system_exact_peak','model_parameter_bytes':MODEL_BYTES[model]}
        for key in group[0]['build_ram']:
            values=[r['build_ram'][key] for r in group]
            if len(set(values))!=1: raise GateError('Build RAM size changed: '+key)
            row[key]=values[0]
        for field,mode in summary_fields(kind).items():
            values=[integer(r['result']['ram'],field) for r in group]
            if mode=='constant':
                if len(set(values))!=1: raise GateError('Inconsistent RAM constant: '+field)
                row[field]=values[0]
            elif mode=='max': row[field+'_maximum_observed']=max(values)
            elif mode=='min': row[field+'_minimum_observed']=min(values)
        output.append(row)
    return {'schema':'kanids-ram500-summary-v1','board_kind':kind,'complete':True,'runs':15,'rows':output,
            'aggregation':'maximum_observed_or_minimum_headroom_across_three_boots_no_mean_peak',
            'limitations':['Workload observation, not a universal worst-case bound',
                           'Diagnostic firmware differs from the accepted energy firmware',
                           'Stack watermarks count touched bytes; reserved but untouched frames need compiler evidence',
                           'C3 allocated task stacks are already included in allocated heap; do not add them twice']}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',required=True,choices=tuple(BOARDS)); parser.add_argument('--port')
    parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--bootstrap-evidence',type=Path)
    args=parser.parse_args(argv); root=Path(__file__).resolve().parent; project=root/'project'; kind=args.board
    directory=None; suite=None; code=1
    try:
        entries=verify_sources(root)
        if args.dry_run:
            print(json.dumps({'protocol':protocol_for(kind),'board':BOARDS[kind],'plan':plan_for(kind),'hardware_access':False,
                              'meter_needed':False,'dependencies':PINS,'source_files_verified':len(entries)},ensure_ascii=False,indent=2))
            print('DRY RUN PASS. Подключите выбранную плату напрямую по USB. FNB58 и его приложение не нужны.')
            return 0
        sid=datetime.now(timezone.utc).strftime(kind.upper()+'_RAM500_%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]
        directory=root/'runs'/sid; directory.mkdir(parents=True,exist_ok=False)
        suite={'schema':'kanids-ram500-suite-v1','suite_id':sid,'board_kind':kind,'expected_board':BOARDS[kind],
               'started_utc':utc(),'status':'incomplete','source_manifest':digest(root/'KIT_MANIFEST.json'),
               'protocol_file':digest(root/'PROTOCOL.json'),'python':sys.version,'system':platform.platform(),
               'plan':plan_for(kind),'builds':{},'attempts':[],'meter_required':False}
        copy_sources(root,directory/'sources',entries)
        suite['versions']=check_versions()
        import serial
        from serial.tools import list_ports
        if args.bootstrap_evidence:
            evidence=args.bootstrap_evidence.resolve(); verify_packet(evidence)
            shutil.copytree(evidence,directory/'bootstrap'/evidence.name)
        if not args.build_only: suite['board_observed']=check_board(list_ports.comports(),kind,args.port)
        write_json(directory/'SUITE_RECORD.json',suite)
        for model in MODELS:
            destination=directory/'builds'/model; destination.mkdir(parents=True,exist_ok=False)
            build={'model':model,'environment':env_name(kind,model),'commands':[]}
            try:
                run_command([sys.executable,'-m','platformio','run','-d',str(project),'-e',env_name(kind,model)],destination,'01_build',build)
                run_command([sys.executable,'-m','platformio','pkg','list','-d',str(project),'-e',env_name(kind,model)],destination,'02_packages',build,timeout=120)
                verify_sources(root)
                build['binaries'],build['audit']=copy_binaries(project,kind,model,destination/'binaries')
            finally:
                write_json(destination/'BUILD_RECORD.json',build)
            suite['builds'][model]=build; write_json(directory/'SUITE_RECORD.json',suite)
        if args.build_only:
            suite['status']='all_five_builds_pass_no_upload'; code=0; return code
        records=run_plan(root,project,directory,suite,serial,list_ports.comports)
        summary=make_summary(records,kind)
        write_json(directory/'RAM_SUMMARY.json',summary)
        with (directory/'RAM_SUMMARY.csv').open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary['rows'][0])); writer.writeheader(); writer.writerows(summary['rows'])
        suite['status']='all_15_runs_accepted'; code=0
        print('\nВсе 15 опытов RAM завершены. Пришлите итоговый ZIP, указанный ниже.')
    except (Exception,KeyboardInterrupt) as exc:
        error=type(exc).__name__+': '+str(exc)
        if suite is not None: suite['error']=error
        print('\nОСТАНОВКА: '+error+'\nСледующие опыты не запускались. Сохраните ZIP логов.',file=sys.stderr)
    finally:
        if directory is not None and suite is not None:
            suite['finished_utc']=utc(); write_json(directory/'SUITE_RECORD.json',suite)
            archive=make_archive(directory)
            print('\nАрхив результатов: '+str(archive),flush=True)
    return code
def validate_build_audit(audit,kind,model):
    if audit.get('schema')!='kanids-ram500-build-v1' or audit.get('environment')!=env_name(kind,model):
        raise GateError('Wrong target build audit')
    for key in ('active_symbol_present','prediction_symbol_present','batch_prediction_call_verified','stack_probe_symbol_present'):
        if audit.get(key) is not True: raise GateError('Target disassembly gate failed: '+key)
    if audit.get('physical_measurement') is not False or audit.get('total_peak_claim') is not False or audit.get('peak_sram_measured') is not False:
        raise GateError('Build audit incorrectly claims a physical/peak measurement')
    if not isinstance(audit.get('stack_usage_files'),int) or audit['stack_usage_files']<1: raise GateError('Missing stack usage evidence')
    static=audit.get('static_sram_bytes'); diag=audit.get('diagnostic_globals_bytes')
    if not isinstance(static,int) or not isinstance(diag,int) or not 0<diag<=static: raise GateError('Invalid static/diagnostic RAM accounting')
    if kind=='mega':
        if not (static<4096 and audit.get('near_progmem_checked') and 0<audit.get('near_progmem_max_end',0)<=65536):
            raise GateError('Mega static SRAM / PROGMEM placement failed')
        if audit.get('avr_paint_guard_bytes')!=16 or audit.get('avr_leaf_paint_scan_stack_bytes')!={'ram_avr_paint':3,'ram_avr_scan':3}:
            raise GateError('AVR painter/scanner leaf stack proof changed')
        codegen=audit.get('avr_heap_control_codegen')
        if not isinstance(codegen,dict) or codegen.get('passed') is not True or codegen.get('failures')!=[]:
            raise GateError('Missing or failed AVR heap-control code generation proof')
        source=audit.get('avr_heap_control_source')
        if not isinstance(source,dict) or source.get('passed') is not True or not isinstance(source.get('checks'),dict):
            raise GateError('Missing or failed AVR heap-control source proof')
        for key in ('heap_growth_128','malloc_counter_increment_one','free_counter_increment_one',
                    'nonnull_probe','allocation_size_128','malloc_delta_measured','free_delta_measured'):
            if source['checks'].get(key) is not True: raise GateError('AVR heap-control source predicate failed: '+key)
    elif not 0<static<327680 or not isinstance(audit.get('iram_code_bytes'),int) or audit['iram_code_bytes']<0:
        raise GateError('Invalid C3 linker RAM accounting')

def validate_metrics(data,kind,audit=None):
    if data.get('ram_valid')!='1' or data.get('total_peak_claim')!='0': raise GateError('Invalid RAM status or unsupported global peak claim')
    row_bytes=32 if data['model']=='dt5' else 24
    if integer(data,'row_buffer_bytes')!=row_bytes: raise GateError('Wrong frozen row buffer size')
    if kind=='mega':
        for key in ('malloc_calls','calloc_calls','realloc_calls','free_calls'):
            if integer(data,key)!=0: raise GateError('Unexpected active AVR heap operation: '+key)
        lo=integer(data,'paint_last_low'); hi=integer(data,'paint_last_high_exclusive')
        before=integer(data,'heap_end_before'); after=integer(data,'heap_end_after')
        if not 512<=before==after==lo<hi<=8703: raise GateError('Invalid AVR heap/paint bounds')
        untouched=integer(data,'paint_min_untouched_bytes'); stack=integer(data,'stack_observed_bytes'); sp=integer(data,'min_sampled_sp')
        if not 0<untouched<=hi-lo or not lo<sp<=8703 or not 0<stack<=8192:
            raise GateError('Invalid AVR stack observation/headroom')
        if stack<8703-sp: raise GateError('AVR stack depth smaller than sampled SP evidence')
        if audit and before<512+audit['static_sram_bytes']: raise GateError('AVR heap overlaps linked static SRAM')
        for key,value in {'paint_guard_bytes':'16','patterns':'165,90,60','stack_scope':'load_predict_check_and_observed_ISR'}.items():
            if data.get(key)!=value: raise GateError('Wrong AVR watermark protocol: '+key)
    else:
        reserved=integer(data,'loop_stack_reserved_bytes')
        before=integer(data,'loop_stack_hwm_before_bytes'); after=integer(data,'loop_stack_hwm_after_bytes'); minimum=integer(data,'loop_stack_hwm_min_bytes')
        if not 0<minimum<=after<=before<=reserved==8192: raise GateError('Invalid C3 task watermark sequence')
        if integer(data,'loop_stack_observed_used_bytes')!=reserved-minimum: raise GateError('C3 task used bytes inconsistent')
        for phase in ('before','after'):
            free=integer(data,'heap_free_'+phase+'_bytes'); allocated=integer(data,'heap_allocated_'+phase+'_bytes')
            largest=integer(data,'heap_largest_'+phase+'_bytes'); low=integer(data,'heap_lifetime_min_'+phase+'_bytes')
            if not 0<low<=free<=327680 or not 0<largest<=free or not 0<allocated<=327680:
                raise GateError('Invalid C3 heap observation: '+phase)
        if integer(data,'heap_lifetime_min_after_bytes')>integer(data,'heap_lifetime_min_before_bytes'):
            raise GateError('Lifetime heap minimum cannot increase')
        if not 0<integer(data,'heap_sampled_min_free_bytes')<=327680: raise GateError('Invalid C3 sampled free heap')
        for key,value in {'heap_domain':'internal_8bit','heap_min_scope':'sum_region_lifetime_minima',
                          'stack_scope':'loop_task_lifetime','stack_inside_heap':'1'}.items():
            if data.get(key)!=value: raise GateError('Wrong C3 heap/stack scope: '+key)

def validate_control_metrics(data,kind,result):
    before=integer(data,'stack_before_bytes'); after=integer(data,'stack_after_bytes')
    hb=integer(data,'heap_before'); hd=integer(data,'heap_during'); ha=integer(data,'heap_after')
    if data.get('baseline_already_saved')!='1': raise GateError('Control polluted baseline measurement')
    if kind=='mega':
        if not (0<=before<after<=8192 and after-before>=128 and 512<=hb<hd<=8703 and hd-hb>=128 and 512<=ha<=hd):
            raise GateError('AVR instrumentation did not observe known probes')
        for key in ('heap_alloc_ok','heap_malloc_calls','heap_free_calls'):
            if integer(data,key)!=1: raise GateError('AVR heap positive control requires exactly one successful malloc/free pair: '+key)
        expected={'stack_probe_bytes':'256','heap_probe_bytes':'128','stack_metric':'observed_depth','heap_metric':'heap_end_address','separate_task':'0'}
        paint_before=integer(data,'paint_stack_before_bytes'); paint_after=integer(data,'paint_stack_after_bytes')
        if not 0<=paint_before<paint_after<=8192 or paint_after-paint_before<128:
            raise GateError('AVR watermark failed separate positive control')
    else:
        if not (0<after<before<=4096 and before-after>=512 and 0<hd<hb<=327680 and hb-hd>=512 and hd+512<=ha<=327680):
            raise GateError('C3 instrumentation did not observe known probes')
        expected={'stack_probe_bytes':'768','heap_probe_bytes':'512','stack_metric':'free_watermark','heap_metric':'free_bytes','separate_task':'1'}
    if any(data.get(k)!=v for k,v in expected.items()): raise GateError('Wrong control configuration')

def summary_fields(kind):
    fields={'row_buffer_bytes':'constant'}
    if kind=='mega':
        fields.update(stack_observed_bytes='max',paint_min_untouched_bytes='min',min_sampled_sp='min',
                      heap_end_before='constant',heap_end_after='constant',malloc_calls='max',calloc_calls='max',realloc_calls='max',free_calls='max')
    else:
        fields.update(loop_stack_reserved_bytes='constant',loop_stack_observed_used_bytes='max',loop_stack_hwm_min_bytes='min',
                      heap_free_before_bytes='min',heap_free_after_bytes='min',heap_allocated_before_bytes='max',heap_allocated_after_bytes='max',
                      heap_largest_before_bytes='min',heap_largest_after_bytes='min',heap_lifetime_min_before_bytes='min',heap_lifetime_min_after_bytes='min',
                      heap_sampled_min_free_bytes='min')
    return fields

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace'); sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    raise SystemExit(main())
