#!/usr/bin/env python3
import argparse, json, hashlib, pathlib, importlib.util, subprocess, re, sys
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description="Revalidate the accepted Mega RAM500 v0.14.1 session from raw records and actual AVR ELF binaries.")
parser.add_argument('--session',type=pathlib.Path,required=True,help="Extracted Mega session directory containing PACKET_MANIFEST.json")
parser.add_argument('--kit',type=pathlib.Path,required=True,help="Original v0.14.1 kit directory containing KIT_MANIFEST.json")
parser.add_argument('--tool-bin',type=pathlib.Path,required=True,help="Pinned AVR toolchain bin directory; .exe executables are supported")
parser.add_argument('--output',type=pathlib.Path,required=True,help="New output directory; must not exist or be inside an input directory")
args=parser.parse_args()
RUN=args.session.resolve(); KIT=args.kit.resolve(); TOOLS=args.tool_bin.resolve(); OUT=args.output.resolve()
for label,path in [('session',RUN),('kit',KIT),('tool-bin',TOOLS)]:
 if not path.is_dir():parser.error('--'+label+' must name an existing directory')
if OUT.exists():parser.error('--output must name a new directory; existing output is never overwritten')
if any(OUT==path or path in OUT.parents for path in [RUN,KIT,TOOLS]):
 parser.error('--output must be outside the session, kit and tool-bin input directories')
def tool(name):
 for candidate in [TOOLS/(name+'.exe'),TOOLS/name]:
  if candidate.is_file():return str(candidate)
 parser.error('Missing AVR executable: '+name+' in '+str(TOOLS))
AVR_OBJDUMP=tool('avr-objdump'); AVR_SIZE=tool('avr-size'); AVR_OBJCOPY=tool('avr-objcopy')
OUT.mkdir(parents=True,exist_ok=False)
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
checks=[]
def check(label,p):
 if not p:raise AssertionError(label)
 checks.append(label)
def manifest(base,fn):
 m=js(base/fn); paths=set()
 for f in m['files']:
  p=base/f['path'];check(fn+': '+f['path'],p.is_file() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256']);paths.add(f['path'])
 return m,paths
packet,paths=manifest(RUN,'PACKET_MANIFEST.json')
check('packet manifest covers all payload files',paths=={p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file() and p.relative_to(RUN).as_posix()!='PACKET_MANIFEST.json'})
sm,sp=manifest(RUN/'sources','KIT_MANIFEST.json')
for n in [*sorted(sp),'KIT_MANIFEST.json']:check('released source identity '+n,(KIT/n).read_bytes()==(RUN/'sources'/n).read_bytes())
runner=mod('ram_runner_review',KIT/'run_ram.py');codegen=mod('codegen_review',KIT/'project/check_avr_control_codegen.py')
suite=js(RUN/'SUITE_RECORD.json');check('15-run complete',suite['status']=='all_15_runs_accepted' and len(suite['attempts'])==15)
check('expected physical board',suite['expected_board']==runner.BOARDS['mega'])
models={};builds={}
for name in runner.MODELS:
 bdir=RUN/'builds'/name/'binaries';b=suite['builds'][name];audit=js(bdir/'firmware.audit.json')
 check('build '+name+' exit codes',all(c['returncode']==0 for c in b['commands']))
 for fn,d in b['binaries'].items():check(name+' binary '+fn,(bdir/fn).stat().st_size==d['bytes'] and sha(bdir/fn)==d['sha256'])
 runner.validate_build_audit(audit,'mega',name)
 actual_dis=subprocess.check_output([AVR_OBJDUMP,'-d','-C',str(bdir/'firmware.elf')],text=True,encoding='utf-8',errors='strict')
 generated=codegen.check_text(actual_dis);check(name+' actual ELF heap-control machine code',generated['passed'])
 (OUT/(name+'_actual_disassembly.txt')).write_text(actual_dis,encoding='utf-8')
 check(name+' supplied disassembly actual instructions',codegen.instructions(actual_dis)==codegen.instructions((bdir/'firmware.disassembly.txt').read_text(encoding='utf-8')))
 size=subprocess.check_output([AVR_SIZE,'-A',str(bdir/'firmware.elf')],text=True,encoding='utf-8',errors='strict')
 sram=sum(int(m.group(1)) for line in size.splitlines() if (m:=re.match(r'^\.(?:data|bss|noinit)\s+(\d+)\b',line)))
 check(name+' actual ELF static SRAM',sram==335==audit['static_sram_bytes'])
 hex_out=OUT/(name+'_from_elf.hex')
 subprocess.run([AVR_OBJCOPY,'-O','ihex','-R','.eeprom',str(bdir/'firmware.elf'),str(hex_out)],check=True)
 check(name+' flashed HEX reproduced from ELF',hex_out.read_bytes()==(bdir/'firmware.hex').read_bytes())
 builds[name]={'static_sram_bytes':sram,'elf_sha256':sha(bdir/'firmware.elf'),'hex_sha256':sha(bdir/'firmware.hex'),'codegen':generated}
 models[name]=[]
for step in suite['plan']:
 key=step['key'];ad=RUN/'attempts'/key;r=js(ad/'RUN_RECORD.json');base=js(ad/'BASELINE_RECORD.json');ser=ad/'serial_connection_00';events=[json.loads(x) for x in (ser/'serial.jsonl').read_text(encoding='utf-8').splitlines()]
 rx=b''.join(bytes.fromhex(e['hex']) for e in events if e['direction']=='RX');tx=[bytes.fromhex(e['hex']).decode('ascii') for e in events if e['direction']=='TX']
 check(key+' UART binary agrees with JSONL',rx==(ser/'serial_raw.bin').read_bytes())
 lines=rx.decode('ascii').splitlines();check(key+' readable UART log',lines==(ser/'serial.log').read_text(encoding='utf-8').splitlines())
 check(key+' exact command sequence',tx==['INFO\n','RUN\n','CONTROL\n'])
 check(key+' ten response lines',len(lines)==10)
 model=step['model'];audit=js(RUN/'builds'/model/'binaries/firmware.audit.json')
 info=runner.validate_info(lines[:6],'mega',model,KIT);result=runner.validate_result(lines[6:9],'mega',model,audit);control=runner.validate_control(lines[9:],'mega',result)
 check(key+' raw decoded records equal records',info==r['info'] and result==r['result'] and control==r['control'])
 check(key+' accepted record',r['status']=='accepted_observed_ram_components' and r['plan_item']==step)
 check(key+' preserved baseline before CONTROL',base['result']==result and 'control' not in base and r['baseline_end_utc']<=r['control_start_utc'])
 check(key+' board before/after upload',all(r[k]['serial_number']==runner.BOARDS['mega']['serial'] and r[k]['vid']==9025 and r[k]['pid']==66 for k in ['board_before_upload','board_after_upload']))
 check(key+' fresh verified upload',r['fresh_boot_method']=='upload_each_repeat' and len(r['commands'])==1 and r['commands'][0]['returncode']==0 and '-t' in r['commands'][0]['argv'] and 'nobuild' in r['commands'][0]['argv'] and 'upload' in r['commands'][0]['argv'])
 upload=(ad/'03_upload.log').read_text(encoding='utf-8');check(key+' device signature and flash verification',re.search(r'Device signature = 0x1e9801',upload) is not None and re.search(r'\d+ bytes of flash verified',upload) is not None and '[SUCCESS]' in upload)
 check(key+' upload bound to saved binary',r['build_identities']==suite['builds'][model]['binaries'])
 d=result['ram'];c=control
 check(key+' independent counters',int(d['checked'])==1500 and int(d['mismatches'])==0 and int(d['checksum'])==3*runner.EXPECTED_SUMS[model] and int(d['heap_end_before'])==int(d['heap_end_after'])==847)
 check(key+' independent allocator controls',int(c['heap_before'])==847 and int(c['heap_during'])==977 and int(c['heap_after'])==847 and all(int(c[k])==1 for k in ['heap_pass','heap_alloc_ok','heap_malloc_calls','heap_free_calls']))
 check(key+' independent stack controls',int(c['stack_after_bytes'])-int(c['stack_before_bytes'])>=128 and int(c['paint_stack_after_bytes'])-int(c['paint_stack_before_bytes'])>=128)
 models[model].append({'key':key,'repeat':r['repeat'],'stack_observed_bytes':int(d['stack_observed_bytes']),'untouched_bytes':int(d['paint_min_untouched_bytes']),'min_sampled_sp':int(d['min_sampled_sp']),'row_buffer_bytes':int(d['row_buffer_bytes']),'control_stack_before':int(c['stack_before_bytes']),'control_stack_after':int(c['stack_after_bytes']),'heap_before':int(c['heap_before']),'heap_during':int(c['heap_during']),'heap_after':int(c['heap_after']),'checked':int(d['checked'])})
summary=[]
for name,rows in models.items():
 check(name+' three distinct uploads',sorted(r['repeat'] for r in rows)==[1,2,3])
 summary.append({'model':name,'static_sram_bytes':335,'observed_stack_depths_bytes':[r['stack_observed_bytes'] for r in rows],'maximum_observed_stack_bytes':max(r['stack_observed_bytes'] for r in rows),'minimum_untouched_bytes':min(r['untouched_bytes'] for r in rows),'row_buffer_bytes':rows[0]['row_buffer_bytes'],'active_heap_calls':0,'checked':sum(r['checked'] for r in rows)})
recomputed=runner.make_summary([js(RUN/'attempts'/step['key']/'RUN_RECORD.json') for step in suite['plan']],'mega')
check('summary reproduced from accepted records',recomputed==js(RUN/'RAM_SUMMARY.json'))
report={'schema':'independent-mega-ram141-acceptance-v1','accepted':True,'packet_manifest_files':len(paths),'released_source_files_plus_manifest':len(sp)+1,'checks_passed':len(checks),'runs':15,'predictions_checked':22500,'mismatches':0,'builds':builds,'summary':summary,'per_run':models,'checks':checks,'limitations':['Observed finite-workload stack depths, not a proof of worst-case global Peak RAM.','All five diagnostic ELF images have 335 B static SRAM; row buffers and diagnostic globals are included there and must not be added twice.','Instrumented firmware differs from previous latency and energy firmware.','Stack baseline includes active workload and observed ISR activity; positive-control stack and heap allocations occur afterward and are not workload results.','All 1500 checks per boot compare against frozen reference predictions, not 1500 independent new flows.']}
(OUT/'MEGA_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['accepted','packet_manifest_files','released_source_files_plus_manifest','checks_passed','runs','predictions_checked','mismatches','summary']},indent=2))
