from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_suite as r


def port(name='COM5',kind='mega',serial=None):
    b=r.BOARDS[kind]
    return SimpleNamespace(device=name,vid=b['vid'],pid=b['pid'],serial_number=serial or b['serial'],description='test')


def ready_lines(model='coeff'):
    return ['CORRECTNESS checked=500 correct=1 reference=compiled_C_predictions',
            'READY count=12000 target_us=120000000 cal_n=500 cal_us=5000000 checksum_per500='+str(r.EXPECTED_SUMS[model])+' warmup_count=500']


def done_lines():
    values=[('pre_led_on_1',0),('pre_led_off_1',2000000),('pre_led_on_2',4000000),('pre_led_off_2',8000000),('pre_led_on_3',10000000),('pre_led_off_3',12000000),('idle_before_begin',12000008),('idle_before_end',22000008),('active_begin',22000008),('active_end',142000008),('idle_after_begin',142000008),('idle_after_end',152000008),('post_led_on_1',152000016),('post_led_off_1',154000016),('post_led_on_2',156000016),('post_led_off_2',160000016),('post_led_on_3',162000016),('post_led_off_3',164000016)]
    return ['PREP sync=begin','DONE count=12000 active_us=120000000 checksum=6144 expected=6144 correct=1 timing_ok=1 wdt_restored=1']+['EVENT name='+n+' us='+str(t) for n,t in values]+['WAIT_DONE reset_required=0 rearm_allowed=1']


def info_lines(kind='mega',model='coeff'):
    common=['HELLO protocol='+r.PROTOCOL,'MODEL variant='+model+' model_bytes='+str(r.MODEL_BYTES[model])+' cohort_sha256='+r.COHORT_SHA,
            'COHORT rows=500 order=attack_normal_interleaved boundary=flash_row_load_predict_checksum model_placement=flash',
            'RAW_IDS values='+','.join(map(str,r.read_json(ROOT/'FROZEN_COHORT_REFERENCE.json')['row_ids']))]
    if kind=='mega':
        common+=['SYSTEM chip=ATmega2560 cpu_mhz=16 flash_bytes=262144 timer=micros timer_bits=32 elapsed=unsigned_modulo interrupts=enabled active_yield=0 led_gpio=13 led_active=HIGH led_confirmed=0','WDT policy=unchanged_not_instrumented']
    else:
        common+=['SYSTEM chip=ESP32-C3 mac=10:00:3B:CB:8D:70 cpu_mhz=160 flash_bytes=4194304 timer=esp_timer_get_time timer_bits=64 interrupts=enabled active_yield=0 led_gpio=8 led_active=LOW led_confirmed=0 wifi=not_initialized bt=not_initialized','WDT policy=idle0_subscription_temporarily_suspended idle_before=1 loop_before=0']
    return common+['INFO_DONE state=idle']


def cfn(path,n=1701,rate=10,start=0):
    raw=bytearray(struct.pack('<diiih',rate,start,0,5,3))
    for code in (0,1,4): raw.extend(struct.pack('<hIB',code,0,0))
    raw.extend(struct.pack('<i',n))
    for i in range(n): raw.extend(struct.pack('<dddd',i/rate,5,.07,.35))
    path.write_bytes(raw)


class BoardIdentity(unittest.TestCase):
    def test_only_exact_board_even_with_meter(self):
        meter=SimpleNamespace(device='COM11',vid=0x2E3C,pid=0x5558,serial_number='0C4301EC3C64',description='meter')
        self.assertEqual(r.check_board([meter,port()],'mega')['port'],'COM5')
    def test_meter_requested_rejected(self):
        with self.assertRaises(r.GateError): r.check_board([port()],'mega','COM11')
    def test_spare_c3_rejected(self):
        with self.assertRaises(r.GateError): r.check_board([port('COM12','c3','10:00:3B:CC:2C:C4')],'c3')
    def test_c3_renumbering_allowed_exact_serial(self):
        self.assertEqual(r.check_board([port('COM18','c3')],'c3')['port'],'COM18')
    def test_two_matches_rejected(self):
        with self.assertRaises(r.GateError): r.check_board([port(),port('COM9')],'mega')
    def test_serial_match_without_vid_is_rejected(self):
        p=port();p.vid=0xFFFF
        with self.assertRaises(r.GateError):r.check_board([p],'mega')

class Plan(unittest.TestCase):
    def test_balanced_positions_and_complete_blocks(self):
        p=r.plan_for('mega');self.assertEqual(len(p),26);self.assertEqual(p[0]['phase'],'pilot')
        for block in range(5): self.assertEqual({x['model'] for x in p[1+block*5:6+block*5]},set(r.MODELS))
        for position in range(5):self.assertEqual({p[1+b*5+position]['model'] for b in range(5)},set(r.MODELS))
    def test_dry_run_no_hardware_or_source_reads(self):
        with patch('run_suite.verify_sources',side_effect=AssertionError('No source I/O')),patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(r.main(['--board','mega','--dry-run']),0)
        self.assertFalse(json.loads(out.getvalue())['hardware_access'])

class Protocol(unittest.TestCase):
    def test_realistic_micros_boundary_overhead(self):
        ready=r.validate_ready(ready_lines(),'coeff'); result=r.validate_done(done_lines(),ready,'mega')
        self.assertEqual(result['done']['checksum'],6144)
    def test_all_five_expected_sums(self):
        for model in r.MODELS:self.assertEqual(r.validate_ready(ready_lines(model),model)['checksum_per500'],r.EXPECTED_SUMS[model])
    def test_self_consistent_wrong_model_sum_rejected(self):
        lines=ready_lines();lines[1]=lines[1].replace('checksum_per500=256','checksum_per500=250')
        with self.assertRaises(r.GateError):r.validate_ready(lines,'coeff')
    def test_count_must_follow_calibration(self):
        lines=ready_lines();lines[1]=lines[1].replace('count=12000','count=12500',1)
        with self.assertRaises(r.GateError):r.validate_ready(lines,'coeff')
    def test_missing_golden_check_rejected(self):
        with self.assertRaises(r.GateError):r.validate_ready(ready_lines()[1:],'coeff')
    def test_wrong_checksum_rejected(self):
        lines=done_lines();lines[1]=lines[1].replace('checksum=6144','checksum=6143')
        with self.assertRaises(r.GateError):r.validate_done(lines,r.validate_ready(ready_lines(),'coeff'),'mega')
    def test_duplicate_event_rejected(self):
        lines=done_lines();lines.insert(-1,lines[2])
        with self.assertRaises(r.GateError):r.validate_done(lines,r.validate_ready(ready_lines(),'coeff'),'mega')
    def test_c3_wdt_receipts_required(self):
        with self.assertRaises(r.GateError):r.validate_done(done_lines(),r.validate_ready(ready_lines(),'coeff'),'c3')
    def test_c3_wdt_restored_checked(self):
        lines=['WDT phase=before idle_before=1 loop_before=0 suspended=1']+done_lines()
        lines.insert(-1,'WDT phase=after idle_before=1 loop_before=0 suspended=1 restored=1')
        r.validate_done(lines,r.validate_ready(ready_lines(),'coeff'),'c3')
        lines[-2]=lines[-2].replace('restored=1','restored=0')
        with self.assertRaises(r.GateError):r.validate_done(lines,r.validate_ready(ready_lines(),'coeff'),'c3')
    def test_valid_info_both_boards(self):
        for kind in ('mega','c3'):r.validate_info(info_lines(kind),kind,'coeff')
    def test_wrong_row_order_rejected(self):
        lines=info_lines();lines[3]=lines[3].replace('57978,126454','126454,57978')
        with self.assertRaises(r.GateError):r.validate_info(lines,'mega','coeff')
    def test_host_simulation_never_hardware_accepted(self):
        lines=info_lines();lines.insert(1,'EXECUTION mode=host_simulation hardware_measurement=0')
        with self.assertRaises(r.GateError):r.validate_info(lines,'mega','coeff')
    def test_c3_wifi_on_rejected(self):
        lines=info_lines('c3');lines[4]=lines[4].replace('wifi=not_initialized','wifi=initialized')
        with self.assertRaises(r.GateError):r.validate_info(lines,'c3','coeff')

class Trace(unittest.TestCase):
    def test_complete_structural_trace_and_duplicate_reject(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trace.cfn';cfn(p);a=r.validate_cfn(p,120000000,set())
            self.assertEqual(a['metadata']['records'],1701)
            with self.assertRaises(r.GateError):r.validate_cfn(p,120000000,{a['file']['sha256']})
    def test_truncated_trace_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trace.cfn';cfn(p);p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):r.validate_cfn(p,120000000,set())
    def test_wrong_rate_and_short_trace_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'trace.cfn';cfn(p,rate=100)
            with self.assertRaises(r.GateError):r.validate_cfn(p,120000000,set())
            cfn(p,n=1000)
            with self.assertRaises(r.GateError):r.validate_cfn(p,120000000,set())
    def test_invalid_trace_is_preserved_before_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);attempt=root/'attempt';attempt.mkdir();p=root/'bad.cfn';p.write_bytes(b'invalid')
            record={'run_command_utc':r.utc(),'result':{'done':{'active_us':120000000}}}
            with self.assertRaises(ValueError):r.collect_cfn(attempt,p,record,set(),prompt=lambda _: '')
            self.assertEqual((attempt/p.name).read_bytes(),b'invalid')
            self.assertTrue((attempt/'RUN_RECORD.json').is_file())

class Archive(unittest.TestCase):
    def test_nested_manifest_preserved_and_resume_verifiable(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'session';attempt=root/'attempts'/'one';attempt.mkdir(parents=True)
            r.write_json(attempt/'RUN_RECORD.json',{'status':'test'});r.make_archive(attempt)
            archive=r.make_archive(root)
            target=Path(td)/'extracted'
            with zipfile.ZipFile(archive) as zf:
                self.assertIsNone(zf.testzip());zf.extractall(target)
            r.verify_packet(target/'session'/'attempts'/'one')
            r.verify_packet(target/'session')
    def test_changed_packet_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'session';root.mkdir();(root/'a.txt').write_text('original');r.make_archive(root)
            (root/'a.txt').write_text('edited')
            with self.assertRaises(r.GateError):r.verify_packet(root)
    def test_extra_packet_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'session';root.mkdir();r.make_archive(root);(root/'extra').write_text('extra')
            with self.assertRaises(r.GateError):r.verify_packet(root)


class Resume(unittest.TestCase):
    def create_session(self, root, accepted=True):
        kit=root/'kit';kit.mkdir();(kit/'KIT_MANIFEST.json').write_text('{}');(kit/'PROTOCOL.json').write_text('{}')
        session=root/'session';session.mkdir();attempt=session/'attempts'/'pilot_a';attempt.mkdir(parents=True)
        status=r.ACCEPTED if accepted else 'incomplete'
        data={'plan_item':r.plan_for('mega')[0],'status':status,'cfn_supplied':{'sha256':'tracehash'},'cfn':{'file':{'sha256':'tracehash'}}}
        r.write_json(attempt/'RUN_RECORD.json',data);r.make_archive(attempt)
        record={'schema':'kanids-hw500-suite-v1','board_kind':'mega','plan':r.plan_for('mega'),
                'source_manifest':r.digest(kit/'KIT_MANIFEST.json'),'protocol_file':r.digest(kit/'PROTOCOL.json'),'versions':{'test':'fixed'},
                'attempts':[{'relative_path':'attempts/pilot_a','record_file':r.digest(attempt/'RUN_RECORD.json'),'status':status}]}
        r.write_json(session/'SUITE_RECORD.json',record)
        return kit,session,attempt
    def test_accepted_plan_slots_recovered(self):
        with tempfile.TemporaryDirectory() as td,patch('run_suite.check_versions',return_value={'test':'fixed'}),patch('run_suite.verify_sources'):
            kit,session,attempt=self.create_session(Path(td))
            _,accepted,used=r.verify_resume(kit,session,'mega')
            self.assertEqual(set(accepted),{'pilot_coeff'});self.assertEqual(used,{'tracehash'})
    def test_failed_trace_hash_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as td,patch('run_suite.check_versions',return_value={'test':'fixed'}),patch('run_suite.verify_sources'):
            kit,session,attempt=self.create_session(Path(td),accepted=False)
            _,accepted,used=r.verify_resume(kit,session,'mega')
            self.assertFalse(accepted);self.assertEqual(used,{'tracehash'})
    def test_changed_protocol_refuses_resume(self):
        with tempfile.TemporaryDirectory() as td,patch('run_suite.check_versions',return_value={'test':'fixed'}),patch('run_suite.verify_sources'):
            kit,session,attempt=self.create_session(Path(td));(kit/'PROTOCOL.json').write_text('{"changed":true}')
            with self.assertRaises(r.GateError):r.verify_resume(kit,session,'mega')
    def test_unregistered_crash_attempt_refuses_automatic_resume(self):
        with tempfile.TemporaryDirectory() as td,patch('run_suite.check_versions',return_value={'test':'fixed'}),patch('run_suite.verify_sources'):
            kit,session,attempt=self.create_session(Path(td));(session/'attempts'/'orphan').mkdir()
            with self.assertRaises(r.GateError):r.verify_resume(kit,session,'mega')
    def test_changed_saved_receipt_refuses_resume(self):
        with tempfile.TemporaryDirectory() as td,patch('run_suite.check_versions',return_value={'test':'fixed'}),patch('run_suite.verify_sources'):
            kit,session,attempt=self.create_session(Path(td));(attempt/'RUN_RECORD.json').write_text('{}')
            with self.assertRaises(r.GateError):r.verify_resume(kit,session,'mega')

if __name__=='__main__':unittest.main()
