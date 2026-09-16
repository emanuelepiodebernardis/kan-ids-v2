"""Contract/counter/identity/evidence regressions; no real USB is ever opened."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_ram as r

def line(tag,data): return tag+' '+' '.join(str(k)+'='+str(v) for k,v in data.items())
def audit(kind='mega',model='coeff'):
 return {'schema':'kanids-ram500-build-v1','environment':r.env_name(kind,model),'active_symbol_present':True,
         'prediction_symbol_present':True,'batch_prediction_call_verified':True,'stack_probe_symbol_present':True,
         'physical_measurement':False,'total_peak_claim':False,'peak_sram_measured':False,'stack_usage_files':1,
         'static_sram_bytes':600 if kind=='mega' else 12000,'diagnostic_globals_bytes':160,
         'near_progmem_checked':[{'end':40000}],'near_progmem_max_end':40000,'iram_code_bytes':40000,
         'avr_paint_guard_bytes':16,'avr_leaf_paint_scan_stack_bytes':{'ram_avr_paint':3,'ram_avr_scan':3},
         'avr_heap_control_codegen':{'passed':True,'failures':[]},
         'avr_heap_control_source':{'passed':True,'checks':{key:True for key in (
             'heap_growth_128','malloc_counter_increment_one','free_counter_increment_one','nonnull_probe',
             'allocation_size_128','malloc_delta_measured','free_delta_measured')}}}
def metrics(kind='mega',model='coeff'):
 result={'board':kind,'model':model,'rows':500,'passes':3,'checked':1500,'mismatches':0,
         'checksum':3*r.EXPECTED_SUMS[model],'checksum_per500':r.EXPECTED_SUMS[model],
         'correct':1,'ram_valid':1,'row_buffer_bytes':32 if model=='dt5' else 24,'total_peak_claim':0}
 if kind=='mega':
  result.update(heap_end_before=1112,heap_end_after=1112,malloc_calls=0,calloc_calls=0,realloc_calls=0,free_calls=0,
                stack_observed_bytes=450,min_sampled_sp=8400,paint_min_untouched_bytes=6700,paint_last_low=1112,
                paint_last_high_exclusive=8300,paint_guard_bytes=16,patterns='165,90,60',stack_scope='load_predict_check_and_observed_ISR')
 else:
  result.update(loop_stack_reserved_bytes=8192,loop_stack_hwm_before_bytes=7100,loop_stack_hwm_after_bytes=6600,
                loop_stack_hwm_min_bytes=6600,loop_stack_observed_used_bytes=1592,heap_free_before_bytes=210000,
                heap_free_after_bytes=210000,heap_allocated_before_bytes=30000,heap_allocated_after_bytes=30000,
                heap_largest_before_bytes=110000,heap_largest_after_bytes=110000,heap_lifetime_min_before_bytes=209000,
                heap_lifetime_min_after_bytes=209000,heap_sampled_min_free_bytes=210000,heap_domain='internal_8bit',
                heap_min_scope='sum_region_lifetime_minima',stack_scope='loop_task_lifetime',stack_inside_heap=1)
 return result

def result_lines(kind='mega',model='coeff',run_token=None):
 values=metrics(kind,model);done='DONE correct=1 reset_required=1'
 if run_token is not None:values['run_token']=run_token;done+=' run_token='+run_token
 return ['BEGIN rows=500 passes=3 scope=load_predict_reference_check',line('RAM_RESULT',values),done]
def control(kind='mega',run_token=None):
 values=dict(stack_pass=1,heap_pass=1,correct=1,baseline_already_saved=1)
 if kind=='mega':
  values.update(stack_before_bytes=80,stack_after_bytes=350,heap_before=1112,heap_during=1242,heap_after=1112,
                paint_stack_before_bytes=80,paint_stack_after_bytes=350,stack_probe_bytes=256,heap_probe_bytes=128,
                stack_metric='observed_depth',heap_metric='heap_end_address',separate_task=0,
                heap_alloc_ok=1,heap_malloc_calls=1,heap_free_calls=1)
 else:
  values.update(stack_before_bytes=3800,stack_after_bytes=2900,heap_before=210000,heap_during=209480,heap_after=210000,
                stack_probe_bytes=768,heap_probe_bytes=512,stack_metric='free_watermark',heap_metric='free_bytes',separate_task=1)
 if run_token is not None:values['run_token']=run_token
 return line('CONTROL',values)

def info(kind='mega',model='coeff',root=ROOT):
 system={'chip':r.BOARDS[kind]['chip'],'cpu_mhz':r.BOARDS[kind]['cpu_mhz'],'wdt':'unchanged'}
 if kind=='mega': system.update(sram_capacity_bytes=8192,interrupts='enabled')
 else: system.update(mac=r.BOARDS[kind]['serial'],idf='v4.4.7',loop_stack_reserved_bytes=8192,
                     heap_domain='internal_8bit',watermark_units='bytes',wifi='not_initialized',bt='not_initialized')
 return [line('HELLO',{'protocol':r.protocol_for(kind)}),line('MODEL',{'variant':model,'model_bytes':r.MODEL_BYTES[model],'cohort_sha256':r.COHORT_SHA}),
         'COHORT rows=500 order=attack_normal_interleaved boundary=flash_row_load_predict_checksum model_placement=flash',
         line('RAW_IDS',{'values':','.join(map(str,r.read_json(root/'FROZEN_COHORT_REFERENCE.json')['row_ids']))}),
         line('SYSTEM',system),'INFO_DONE state=idle']
def record(item):
 kind=item['board_kind']; model=item['model']; build=audit(kind,model)
 return {'plan_item':item,'status':r.ACCEPTED,'model':model,'result':r.validate_result(result_lines(kind,model),kind,model,build),
         'control':r.validate_control([control(kind)],kind,{}),
         'build_ram':{k:build[k] for k in ('static_sram_bytes','diagnostic_globals_bytes')}}

# Actual Mega serial line from MEGA_RAM500_20260916T135320Z_3b3703f1,
# repeat01_coeff. Preserved verbatim as evidence of the v0.14.0 failure.
FAILED_V014_CONTROL=('CONTROL stack_pass=1 heap_pass=0 correct=0 stack_before_bytes=0 stack_after_bytes=292 '
 'heap_before=847 heap_during=847 heap_after=847 paint_stack_before_bytes=0 paint_stack_after_bytes=292 '
 'stack_probe_bytes=256 heap_probe_bytes=128 stack_metric=observed_depth heap_metric=heap_end_address '
 'separate_task=0 baseline_already_saved=1')

class ProtocolTests(unittest.TestCase):
 def test_all_ten_contracts(self):
  for kind in r.BOARDS:
   for model in r.MODELS:
    r.validate_info(info(kind,model),kind,model)
    r.validate_result(result_lines(kind,model),kind,model,audit(kind,model))
    r.validate_control([control(kind)],kind,{})
 def test_wrong_board_port(self):
  b=r.BOARDS['c3']; good=SimpleNamespace(device='COM10',vid=b['vid'],pid=b['pid'],serial_number=b['serial'],description='C3')
  other=SimpleNamespace(device='COM11',vid=0x2e3c,pid=0x5558,serial_number='meter',description='meter')
  self.assertEqual(r.check_board([good,other],'c3')['port'],'COM10')
  for ports,requested in [([other],None),([good,other],'COM11'),([good,good],None)]:
   with self.assertRaises(r.GateError): r.check_board(ports,'c3',requested)
 def test_wrong_physical_c3(self):
  lines=info('c3'); lines[-2]=lines[-2].replace(r.BOARDS['c3']['serial'],'10:00:3B:CC:2C:C4')
  with self.assertRaises(r.GateError): r.validate_info(lines,'c3','coeff')
 def test_stale_finished_boot(self):
  lines=info();lines[-1]='INFO_DONE state=done'
  with self.assertRaises(r.GateError): r.validate_info(lines,'mega','coeff')
 def test_missing_or_wrong_order_id(self):
  lines=info();lines[3]=lines[3].replace('values=','values=999999,')
  with self.assertRaises(r.GateError):r.validate_info(lines,'mega','coeff')
 def test_radio_gate(self):
  lines=info('c3');lines[-2]=lines[-2].replace('wifi=not_initialized','wifi=running')
  with self.assertRaises(r.GateError):r.validate_info(lines,'c3','coeff')
 def test_run_counters_each(self):
  for key,new in [('rows',499),('passes',2),('checked',1499),('mismatches',1),('checksum',767),('checksum_per500',257),('correct',0),('ram_valid',0)]:
   vals=metrics();vals[key]=new;lines=result_lines();lines[1]=line('RAM_RESULT',vals)
   with self.subTest(key=key),self.assertRaises(r.GateError):r.validate_result(lines,'mega','coeff',audit())
 def test_sum_alone_insufficient(self):
  vals=metrics();vals.pop('mismatches');lines=result_lines();lines[1]=line('RAM_RESULT',vals)
  with self.assertRaises(r.GateError):r.validate_result(lines,'mega','coeff')
 def test_run_order_and_begin(self):
  lines=result_lines()
  for bad in [lines[1:],lines[::-1],['BEGIN rows=500 passes=3 scope=predict_only']+lines[1:],lines+lines[-1:]]:
   with self.assertRaises(r.GateError):r.validate_result(bad,'mega','coeff')
 def test_heap_call_detected(self):
  lines=result_lines();lines[1]=lines[1].replace('malloc_calls=0','malloc_calls=1')
  with self.assertRaises(r.GateError):r.validate_result(lines,'mega','coeff')
 def test_false_global_peak_claim(self):
  lines=result_lines('c3');lines[1]=lines[1].replace('total_peak_claim=0','total_peak_claim=1')
  with self.assertRaises(r.GateError):r.validate_result(lines,'c3','coeff')
 def test_c3_stack_arithmetic(self):
  lines=result_lines('c3');lines[1]=lines[1].replace('loop_stack_observed_used_bytes=1592','loop_stack_observed_used_bytes=123')
  with self.assertRaises(r.GateError):r.validate_result(lines,'c3','coeff')
 def test_control_numeric_not_selfreport(self):
  for kind in r.BOARDS:
   values=r.fields(control(kind),'CONTROL');values['heap_during']=values['heap_before']
   with self.assertRaises(r.GateError):r.validate_control([line('CONTROL',values)],kind,{})
 def test_actual_v014_failed_control_reports_observations(self):
  with self.assertRaises(r.GateError) as caught:r.validate_control([FAILED_V014_CONTROL],'mega',{})
  for text in ('board=mega','stack_pass=1','heap_pass=0','heap_before=847','heap_during=847','heap_after=847'):
   self.assertIn(text,str(caught.exception))
 def test_actual_v014_failed_control_cannot_be_accepted_by_forging_flags(self):
  values=r.fields(FAILED_V014_CONTROL,'CONTROL')
  values.update(stack_pass='1',heap_pass='1',correct='1')
  with self.assertRaisesRegex(r.GateError,'did not observe known probes'):
   r.validate_control([line('CONTROL',values)],'mega',{})
 def test_nonmoving_heap_rejected_even_with_allocator_evidence(self):
  values=r.fields(FAILED_V014_CONTROL,'CONTROL')
  values.update(stack_pass='1',heap_pass='1',correct='1',heap_alloc_ok='1',heap_malloc_calls='1',heap_free_calls='1')
  with self.assertRaisesRegex(r.GateError,'did not observe known probes'):
   r.validate_control([line('CONTROL',values)],'mega',{})
 def test_control_130_byte_heap_growth_accepted(self):
  values=r.fields(FAILED_V014_CONTROL,'CONTROL')
  values.update(stack_pass='1',heap_pass='1',correct='1',heap_during='977',
                heap_alloc_ok='1',heap_malloc_calls='1',heap_free_calls='1')
  self.assertEqual(r.validate_control([line('CONTROL',values)],'mega',{}),values)
 def test_mega_control_allocator_evidence_required_and_numeric(self):
  for key in ('heap_alloc_ok','heap_malloc_calls','heap_free_calls'):
   for value in (None,'0','2','-1','1.0','yes'):
    values=r.fields(control(),'CONTROL')
    if value is None:values.pop(key)
    else:values[key]=value
    with self.subTest(key=key,value=value),self.assertRaises(r.GateError):
     r.validate_control([line('CONTROL',values)],'mega',{})
 def test_c3_control_does_not_require_avr_allocator_evidence(self):
  values=r.fields(control('c3'),'CONTROL')
  self.assertTrue(all(key not in values for key in ('heap_alloc_ok','heap_malloc_calls','heap_free_calls')))
  self.assertEqual(r.validate_control([line('CONTROL',values)],'c3',{}),values)
 def test_avr_paint_control(self):
  values=r.fields(control(),'CONTROL');values['paint_stack_after_bytes']=values['paint_stack_before_bytes']
  with self.assertRaises(r.GateError):r.validate_control([line('CONTROL',values)],'mega',{})
 def test_build_proof_required(self):
  for key in ('active_symbol_present','prediction_symbol_present','batch_prediction_call_verified','stack_probe_symbol_present'):
   values=audit();values[key]=False
   with self.assertRaises(r.GateError):r.validate_build_audit(values,'mega','coeff')
 def test_avr_flash_bound(self):
  values=audit();values['near_progmem_max_end']=65537
  with self.assertRaises(r.GateError):r.validate_build_audit(values,'mega','coeff')
 def test_avr_heap_codegen_and_source_proofs_mandatory(self):
  for key in ('avr_heap_control_codegen','avr_heap_control_source'):
   for invalid in (None,False,{}, {'passed':False,'failures':[]}):
    values=audit()
    if invalid is None:values.pop(key)
    else:values[key]=invalid
    with self.subTest(key=key,invalid=invalid),self.assertRaises(r.GateError):
     r.validate_build_audit(values,'mega','coeff')
 def test_avr_codegen_failures_override_passed_flag(self):
  for failures in (None,['stale heap snapshot'],'',False):
   values=audit();values['avr_heap_control_codegen']['failures']=failures
   with self.subTest(failures=failures),self.assertRaises(r.GateError):
    r.validate_build_audit(values,'mega','coeff')
 def test_avr_source_predicates_all_mandatory_true(self):
  for key in audit()['avr_heap_control_source']['checks']:
   for invalid in (None,False,1,'true'):
    values=audit()
    if invalid is None:values['avr_heap_control_source']['checks'].pop(key)
    else:values['avr_heap_control_source']['checks'][key]=invalid
    with self.subTest(key=key,invalid=invalid),self.assertRaises(r.GateError):
     r.validate_build_audit(values,'mega','coeff')
 def test_c3_build_needs_no_avr_control_proof(self):
  values=audit('c3');values.pop('avr_heap_control_codegen');values.pop('avr_heap_control_source')
  r.validate_build_audit(values,'c3','coeff')
 def test_complete_mock_15_run_flow(self):
  for kind in r.BOARDS:
   suite={'board_kind':kind,'plan':r.plan_for(kind)}; called=[]
   def fake(root,project,directory,suite,item,serial,ports): called.append(item); return record(item)
   with contextlib.redirect_stdout(io.StringIO()): records=r.run_plan(None,None,None,suite,None,None,acquire=fake)
   self.assertEqual(len(called),15); summary=r.make_summary(records,kind)
   self.assertEqual(len(summary['rows']),5);self.assertTrue(summary['complete'])
 def test_full_acquisition_mock_15_uploads(self):
  for kind in r.BOARDS:
   spec=r.BOARDS[kind]
   port=SimpleNamespace(device='COM99',vid=spec['vid'],pid=spec['pid'],serial_number=spec['serial'],description='mock')
   uploads=[];sent=[]
   class Capture:
    def __init__(self,model,attempt):self.model=model;self.attempt=attempt
    def send(self,command):
     sent.append((self.model,command))
     if command=='CONTROL' or command.startswith('CHECK '):self_test.assertTrue((self.attempt/'BASELINE_RECORD.json').is_file())
    def until(self,terminal,timeout):return result_lines(kind,self.model) if terminal=='DONE' else [control(kind)]
    def c3_frame(self,marker,token,retry_command,baseline_lines=None):
     if marker=='REPORT_BEGIN':return result_lines(kind,self.model,token)
     self_test.assertEqual(baseline_lines,result_lines(kind,self.model,token))
     return [control(kind,token)]
    def c3_verify_tail(self,token,baseline_lines,control_lines):
     self_test.assertEqual(baseline_lines,result_lines(kind,self.model,token))
     self_test.assertEqual(control_lines,[control(kind,token)])
    def close(self):pass
   self_test=self
   def connect(serial,ports,k,model,attempt,root):return SimpleNamespace(close=lambda:None),Capture(model,attempt),{'port':'COM99'},r.validate_info(info(k,model),k,model),[]
   def upload(command,directory,stem,rec,timeout):
    self.assertIn('upload',command);self.assertEqual(command[-1],'COM99');uploads.append(command)
   with tempfile.TemporaryDirectory() as td:
    directory=Path(td);suite={'board_kind':kind,'plan':r.plan_for(kind),'source_manifest':{},'builds':{},'attempts':[]}
    for model in r.MODELS:suite['builds'][model]={'binaries':{},'audit':audit(kind,model)}
    with mock.patch.object(r,'verify_sources',return_value=[]),mock.patch.object(r,'assert_binaries'),mock.patch.object(r,'run_command',side_effect=upload),mock.patch.object(r,'connect_info',side_effect=connect),contextlib.redirect_stdout(io.StringIO()):
     records=r.run_plan(ROOT,ROOT/'project',directory,suite,None,lambda:[port])
    self.assertEqual(len(uploads),15);self.assertEqual(len(suite['attempts']),15)
    self.assertEqual([command.split()[0] for model,command in sent],['RUN','CONTROL' if kind=='mega' else 'CHECK']*15)
    if kind=='c3':
     tokens=[command.split()[1] for model,command in sent]
     self.assertEqual(len(set(tokens)),15)
     self.assertTrue(all(tokens[i]==tokens[i+1] for i in range(0,len(tokens),2)))
    self.assertTrue(r.make_summary(records,kind)['complete'])
    self.assertEqual(len(list((directory/'attempts').glob('*/BASELINE_RECORD.json'))),15)
 def test_partial_no_final_summary(self):
  records=[record(i) for i in r.plan_for('mega')]
  for partial in [records[:-1],records+[records[0]],records[::-1]]:
   with self.assertRaises(r.GateError):r.make_summary(partial,'mega')
 def test_mock_stops_at_failure(self):
  suite={'board_kind':'mega','plan':r.plan_for('mega')};called=[]
  def fake(root,project,directory,suite,item,serial,ports):
   called.append(item)
   if len(called)==4:raise r.GateError('injected failure')
   return record(item)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(r.GateError):r.run_plan(None,None,None,suite,None,None,acquire=fake)
  self.assertEqual(len(called),4)
 def test_maximum_not_average(self):
  records=[record(i) for i in r.plan_for('mega')]
  for index,rec in enumerate(x for x in records if x['model']=='coeff'):rec['result']['ram']['stack_observed_bytes']=str(450+index*20)
  row=r.make_summary(records,'mega')['rows'][0]
  self.assertEqual(row['stack_observed_bytes_maximum_observed'],490)
 def test_packet_detects_changed_raw_log(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'session';root.mkdir();(root/'serial.log').write_text('a')
   r.make_archive(root);r.verify_packet(root);(root/'serial.log').write_text('b')
   with self.assertRaises(r.GateError):r.verify_packet(root)
 def test_duplicate_serial_field(self):
  with self.assertRaises(r.GateError):r.fields('DONE correct=1 correct=1','DONE')
 def test_integer_float_rejected(self):
  with self.assertRaises(r.GateError):r.integer({'n':'1.0'},'n')
 def test_manifest_corrupt_source(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   required=('run_ram.py','bootstrap.py','requirements.txt','START_MEGA.cmd','START_C3.cmd','PROTOCOL.json','project/platformio.ini',
             'project/src/main.cpp','project/save_build_artifacts.py','FROZEN_COHORT_REFERENCE.json','project/include/frozen.h')
   for name in required:
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test')
   r.write_json(root/'project/FROZEN_HEADERS.json',{'files':[{'path':'include/frozen.h',**r.digest(root/'project/include/frozen.h')}]})
   entries=[{'path':p.relative_to(root).as_posix(),**r.digest(p)} for p in root.rglob('*') if p.is_file()]
   r.write_json(root/'KIT_MANIFEST.json',{'files':entries});r.verify_sources(root)
   (root/'project/include/frozen.h').write_text('corrupt')
   with self.assertRaises(r.GateError):r.verify_sources(root)
 def test_upload_encoding_and_stderr_captured(self):
  with tempfile.TemporaryDirectory() as td:
   values={}
   with contextlib.redirect_stdout(io.StringIO()):r.run_command([sys.executable,'-c',"import sys;print('пример ├──');print('stderr marker',file=sys.stderr)"],Path(td),'command',values,timeout=10)
   text=(Path(td)/'command.log').read_text(encoding='utf-8')
   self.assertIn('├──',text);self.assertIn('stderr marker',text)

TOKEN='0123456789abcdef'
def framed_report(token=TOKEN,model='coeff'):
 return ('\nREPORT_BEGIN run_token='+token+'\n'+'\n'.join(result_lines('c3',model,token))+'\n').encode('ascii')
def framed_control(token=TOKEN):
 return ('\nCONTROL_BEGIN run_token='+token+'\n'+control('c3',token)+'\n').encode('ascii')

class MockWire:
 """A byte-fragmented transport; its scripts only reply to host commands."""
 def __init__(self,on_command):self.on_command=on_command;self.chunks=[];self.commands=[]
 @property
 def in_waiting(self):return len(self.chunks[0]) if self.chunks else 0
 def read(self,count):
  if not self.chunks:return b''
  value=self.chunks[0][:count];self.chunks[0]=self.chunks[0][count:]
  if not self.chunks[0]:self.chunks.pop(0)
  return value
 def write(self,data):
  command=data.decode('ascii').strip();self.commands.append(command)
  reply=self.on_command(command)
  if reply:self.chunks.extend(reply if isinstance(reply,list) else [reply])
  return len(data)
 def flush(self):pass

class MockClock:
 def __init__(self):self.now=0
 def __call__(self):self.now+=.01;return self.now

class FramedTransportTests(unittest.TestCase):
 def capture(self,wire,root):return r.SerialCapture(wire,root)
 def test_actual_partial_ram_result_recovers_cached_without_second_run(self):
  partial=('\nRUN_BEGIN run_token='+TOKEN+'\nREPORT_BEGIN run_token='+TOKEN+'\n'+result_lines('c3',run_token=TOKEN)[0]+'\nRAM_RESULT board=c').encode()
  wire=MockWire(lambda command:partial if command.startswith('RUN ') else framed_report())
  with tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
   capture=self.capture(wire,Path(td));capture.send('RUN '+TOKEN)
   lines=capture.c3_frame('REPORT_BEGIN',TOKEN,'RESULT',timeout=.2)
   parsed=r.validate_result(lines,'c3','coeff',audit('c3'),TOKEN)
   self.assertEqual(parsed['ram']['checked'],'1500');capture.close()
   self.assertEqual(wire.commands,['RUN '+TOKEN,'RESULT '+TOKEN])
   state=r.read_json(Path(td)/'TRANSPORT_STATE.json')
   discarded=[event for event in state['recovery_events'] if event['event']=='known_partial_line_discarded']
   self.assertEqual(discarded[0]['line'],'RAM_RESULT board=c')
   self.assertIn(partial,(Path(td)/'serial_raw.bin').read_bytes())
 def test_recovery_inside_every_tag_and_token_prefix(self):
  report=framed_report();cuts=[report.index(b'REPORT_BEGIN')+5,report.index(b'run_token=')+13,
    report.index(b'BEGIN rows')+2,report.index(b'RAM_RESULT')+5,report.index(b'DONE correct')+2,
    report.rindex(b'run_token=')+15]
  for cut in cuts:
   with self.subTest(cut=cut),tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
    wire=MockWire(lambda command:report[:cut] if command.startswith('RUN ') else report)
    capture=self.capture(wire,Path(td));capture.send('RUN '+TOKEN)
    self.assertEqual(capture.c3_frame('REPORT_BEGIN',TOKEN,'RESULT',timeout=.2),result_lines('c3',run_token=TOKEN));capture.close()
    self.assertEqual(wire.commands,['RUN '+TOKEN,'RESULT '+TOKEN])
 def test_wrong_token_reset_and_error_are_fatal_without_retry(self):
  for payload in (framed_report('1111111111111111'),b'HELLO protocol=kanids-ram500-v2\n',b'ESP-ROM:esp32c3\n',b'ERROR reason=no_saved_result\n'):
   with self.subTest(payload=payload[:40]),tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
    wire=MockWire(lambda command:payload);capture=self.capture(wire,Path(td));capture.send('RUN '+TOKEN)
    with self.assertRaises(r.GateError):capture.c3_frame('REPORT_BEGIN',TOKEN,'RESULT',timeout=.2)
    self.assertEqual(wire.commands,['RUN '+TOKEN]);capture.close()
 def test_complete_malformed_frame_never_retried(self):
  bad=framed_report().replace(b'checked=1500',b'checked=1499')
  with tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
   wire=MockWire(lambda command:bad);capture=self.capture(wire,Path(td));capture.send('RUN '+TOKEN)
   lines=capture.c3_frame('REPORT_BEGIN',TOKEN,'RESULT',timeout=.2)
   with self.assertRaises(r.GateError):r.validate_result(lines,'c3','coeff',audit('c3'),TOKEN)
   self.assertEqual(wire.commands,['RUN '+TOKEN]);capture.close()
 def test_missing_done_has_two_bounded_retries_and_retains_partial(self):
  truncated=framed_report().split(b'DONE correct')[0]+b'DO'
  with tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
   wire=MockWire(lambda command:truncated);capture=self.capture(wire,Path(td));capture.send('RUN '+TOKEN)
   with self.assertRaisesRegex(r.TransportTimeout,'after 2 cached reply retries'):
    capture.c3_frame('REPORT_BEGIN',TOKEN,'RESULT',timeout=.2)
   capture.close()
   self.assertEqual(wire.commands,['RUN '+TOKEN,'RESULT '+TOKEN,'RESULT '+TOKEN])
   state=r.read_json(Path(td)/'TRANSPORT_STATE.json');self.assertEqual(state['pending_partial_ascii'],'DO')
   self.assertIn('DO',(Path(td)/'serial.log').read_text())
   self.assertEqual(len([event for event in state['recovery_events'] if event['event']=='frame_timeout']),3)
 def test_result_and_control_tokens_are_bound_to_baseline(self):
  baseline=r.validate_result(result_lines('c3',run_token=TOKEN),'c3','coeff',audit('c3'),TOKEN)
  for wrong in ('1111111111111111',None):
   with self.subTest(wrong=wrong),self.assertRaises(r.GateError):r.validate_control([control('c3',wrong)],'c3',baseline)
  for index in (1,2):
   bad=result_lines('c3',run_token=TOKEN);bad[index]=bad[index].replace(TOKEN,'1111111111111111')
   with self.subTest(index=index),self.assertRaises(r.GateError):r.validate_result(bad,'c3','coeff',audit('c3'),TOKEN)
 def test_check_retry_preserves_baseline_and_does_not_repeat_probe(self):
  with tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
   root=Path(td);baseline=r.validate_result(result_lines('c3',run_token=TOKEN),'c3','coeff',audit('c3'),TOKEN)
   r.write_json(root/'BASELINE_RECORD.json',baseline);before=r.digest(root/'BASELINE_RECORD.json')
   state={'probe_runs':0,'check_requests':0,'saved':None}
   def reply(command):
    self.assertEqual(command,'CHECK '+TOKEN);self.assertEqual(r.digest(root/'BASELINE_RECORD.json'),before)
    state['check_requests']+=1
    if state['saved'] is None:state['probe_runs']+=1;state['saved']=framed_control()
    return state['saved'][:-30] if state['check_requests']==1 else state['saved']
   wire=MockWire(reply);capture=self.capture(wire,root);capture.send('CHECK '+TOKEN)
   lines=capture.c3_frame('CONTROL_BEGIN',TOKEN,'CHECK',timeout=.2,baseline_lines=result_lines('c3',run_token=TOKEN))
   r.validate_control(lines,'c3',baseline);capture.close()
   self.assertEqual(state['probe_runs'],1);self.assertEqual(state['check_requests'],2)
   self.assertEqual(r.digest(root/'BASELINE_RECORD.json'),before)
 def test_late_duplicate_baseline_before_control_must_be_identical(self):
  for corrupt in (False,True):
   with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
    report=framed_report()
    if corrupt:report=report.replace(b'checked=1500',b'checked=1499')
    wire=MockWire(lambda command:report+framed_control());capture=self.capture(wire,Path(td));capture.send('CHECK '+TOKEN)
    if corrupt:
     with self.assertRaisesRegex(r.GateError,'previously complete frame line|differs from saved baseline'):capture.c3_frame('CONTROL_BEGIN',TOKEN,'CHECK',timeout=.5,baseline_lines=result_lines('c3',run_token=TOKEN))
    else:
     self.assertEqual(capture.c3_frame('CONTROL_BEGIN',TOKEN,'CHECK',timeout=.5,baseline_lines=result_lines('c3',run_token=TOKEN)),[control('c3',TOKEN)])
     self.assertTrue(any(e['event']=='identical_cached_baseline_discarded' for e in capture.events))
    capture.close()
 def test_c3_v1_firmware_rejected_mega_v1_unchanged(self):
  old=info('c3');old[0]='HELLO protocol='+r.PROTOCOL
  with self.assertRaises(r.GateError):r.validate_info(old,'c3','coeff')
  r.validate_info(info('mega'),'mega','coeff')
 def test_terminal_tail_accepts_identical_replies_and_rejects_wrong_or_partial(self):
  cases=[(b'',True),(framed_report()+framed_control(),True),(framed_control('1111111111111111'),False),
         (framed_control()[:-3],False),(b'HELLO protocol=kanids-ram500-v2\n',False)]
  for tail,accepted in cases:
   with self.subTest(tail=tail[:40],accepted=accepted),tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
    wire=MockWire(lambda command:b'');wire.chunks=[tail] if tail else []
    capture=self.capture(wire,Path(td))
    if accepted:capture.c3_verify_tail(TOKEN,result_lines('c3',run_token=TOKEN),[control('c3',TOKEN)],quiet_timeout=.2)
    else:
     with self.assertRaises(r.GateError):capture.c3_verify_tail(TOKEN,result_lines('c3',run_token=TOKEN),[control('c3',TOKEN)],quiet_timeout=.2)
    capture.close()
 def test_malformed_complete_prefix_is_fatal_and_cached_prefix_immutable(self):
  full=framed_report();missing_done=full.split(b'DONE correct')[0]
  cases=[(missing_done.replace(b'RAM_RESULT board=c3',b'RAM_RESULT nonsense'),False),
         (missing_done.replace(b'checked=1500',b'checked=1499'),True),
         (missing_done.replace((' run_token='+TOKEN).encode(),b'',1),False)]
  for initial,retried in cases:
   with self.subTest(initial=initial[-100:]),tempfile.TemporaryDirectory() as td,mock.patch.object(r.time,'monotonic',side_effect=MockClock()),contextlib.redirect_stdout(io.StringIO()):
    wire=MockWire(lambda command:initial if command.startswith('RUN ') else full)
    capture=self.capture(wire,Path(td));capture.send('RUN '+TOKEN)
    with self.assertRaises(r.GateError):capture.c3_frame('REPORT_BEGIN',TOKEN,'RESULT',timeout=.2)
    self.assertEqual(len(wire.commands),2 if retried else 1);capture.close()

if __name__=='__main__':unittest.main()
