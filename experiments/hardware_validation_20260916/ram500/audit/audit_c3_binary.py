#!/usr/bin/env python3
"""Read-only standard-library audit. Not a rebuild, hardware run, or formal proof.
python audit_c3_binary.py --session PATH --output NEW_DIRECTORY
"""
import argparse, hashlib, json, re, struct
from pathlib import Path

class ELF:
    def __init__(self, path):
        self.data=path.read_bytes()
        assert self.data[:7] == b'\x7fELF\x01\x01\x01', 'Need ELF32 little-endian'
        h=struct.unpack_from('<HHIIIIIHHHHHH',self.data,16)
        assert h[1] == 243, 'Need RISC-V ELF'
        shoff, shentsize, shnum, shstrndx=h[5],h[10],h[11],h[12]
        self.sections=[]
        for i in range(shnum):
            v=struct.unpack_from('<IIIIIIIIII', self.data, shoff+i*shentsize)
            self.sections.append(dict(zip(['name_offset','type','flags','address','offset','size','link','info','align','entsize'],v)))
        names=self.section_data(self.sections[shstrndx])
        for s in self.sections: s['name']=self.cstr(names,s['name_offset'])
        self.byname={s['name']:s for s in self.sections}
        self.symbols=[]
        for s in self.sections:
            if s['type'] != 2: continue
            strings=self.section_data(self.sections[s['link']])
            for off in range(s['offset'],s['offset']+s['size'],s['entsize']):
                n,val,sz,inf,oth,idx=struct.unpack_from('<IIIBBH',self.data,off)
                self.symbols.append({'name':self.cstr(strings,n),'address':val,'size':sz,'type':inf&15,'bind':inf>>4,'section_index':idx,'section':self.sections[idx]['name'] if 0<idx<len(self.sections) else None})
    @staticmethod
    def cstr(data,offset): return data[offset:data.find(b'\0',offset)].decode('utf-8','replace')
    def section_data(self,s): return self.data[s['offset']:s['offset']+s['size']]
    def read(self,address,size):
        for s in self.sections:
            if s['type'] != 8 and s['flags']&2 and s['address']<=address and address+size<=s['address']+s['size']:
                off=s['offset']+address-s['address'];return self.data[off:off+size]
        raise AssertionError(f'Address unavailable: {address:x}/{size}')

def raw_identifier(name):
    m=re.fullmatch(r'_ZL\d+(ram_\w+)',name)
    return m.group(1) if m else name

def esp_bin(path,elf):
    data=path.read_bytes();assert data[0] == 0xe9
    count=data[1];off=24;segments=[];checksum=0xef
    for i in range(count):
        addr,size=struct.unpack_from('<II',data,off);off+=8
        block=data[off:off+size];assert len(block)==size
        for x in block:checksum^=x
        segments.append({'address':addr,'size':size,'offset':off});off+=size
    checksum_offset=((off+16)&~15)-1
    assert data[checksum_offset]==checksum, 'ESP image XOR checksum'
    image_end=checksum_offset+1
    digest_present=data[23]==1
    if digest_present:
        assert data[image_end:image_end+32]==hashlib.sha256(data[:image_end]).digest(), 'ESP appended SHA256'
        assert len(data)==image_end+32
    matched=[];appdesc_elf_digest=False
    for s in elf.sections:
        if not s['size'] or not s['flags']&2 or s['type']==8:continue
        expected=elf.section_data(s)
        if s['name']=='.flash.appdesc':
            assert expected[144:176]==bytes(32), 'ELF appdesc reserved SHA slot'
            expected=expected[:144]+hashlib.sha256(elf.data).digest()+expected[176:]
            appdesc_elf_digest=True
        cursor=s['address'];end=cursor+s['size'];actual=bytearray()
        while cursor<end:
            seg=next((q for q in segments if q['address']<=cursor<q['address']+q['size']),None)
            assert seg is not None, 'Unmatched allocated ELF section '+s['name']
            count=min(end,seg['address']+seg['size'])-cursor
            pos=seg['offset']+cursor-seg['address'];actual.extend(data[pos:pos+count]);cursor+=count
        assert actual==expected, 'ELF/BIN section '+s['name']
        matched.append(s['name'])
    return {'segment_count':count,'segments':segments,'xor_checksum_valid':True,'appended_sha256_valid':digest_present,'elf_allocated_sections_byte_matched':matched,'appdesc_inserted_elf_sha256_verified':appdesc_elf_digest}

def block(dis,name):
    m=re.search(r'^([0-9a-f]+) <'+re.escape(name)+r'>:\n',dis,re.M);assert m,name
    end=re.search(r'^[0-9a-f]+ <',dis[m.end():],re.M)
    return dis[m.start():m.end()+end.start() if end else len(dis)]

def ordered(text,needles):
    pos=0
    for n in needles:
        pos=text.find(n,pos)
        if pos<0:return False
        pos+=len(n)
    return True

def check_model(session,model):
    b=session/'builds'/model/'binaries';elf=ELF(b/'firmware.elf')
    audit=json.loads((b/'firmware.audit.json').read_text())
    syms={s['name']:s for s in elf.symbols if s['name']}
    nm={m[2]:(int(m[0],16),int(m[1],16)) for m in re.findall(r'^([0-9a-f]+)\s+([0-9a-f]+)\s+\S\s+(\S+)$',(b/'firmware.nm.txt').read_text(),re.M)}
    required=['ram_active_pass','hw500_predict_loaded','ram_stack_probe','_ZL16ram_control_taskPv','_ZL18ram_c3_emit_resultv','_ZL19ram_c3_emit_controlv','_Z4loopv','xIsrStack']
    for n in required:
        assert n in syms and n in nm,n
        assert nm[n]==(syms[n]['address'],syms[n]['size']),n
    static=elf.byname['.dram0.data']['size']+elf.byname['.dram0.bss']['size']
    iram=elf.byname['.iram0.text']['size']+elf.byname['.iram0.text_end']['size']
    assert static==audit['static_sram_bytes'] and iram==audit['iram_code_bytes']
    app_globals=[s for s in elf.symbols if s['type']==1 and raw_identifier(s['name']).startswith('ram_')]
    app_globals.sort(key=lambda s:s['address'])
    for s in app_globals:
        assert s['section'] in ['.dram0.data','.dram0.bss']
        assert nm[s['name']]==(s['address'],s['size'])
    for left,right in zip(app_globals,app_globals[1:]):assert left['address']+left['size']<=right['address'], 'Overlapping globals'
    reported_names={x['symbol'] for x in audit['diagnostic_globals']}
    true_names={s['name'] for s in app_globals}
    assert true_names<=reported_names
    excluded=[syms[n] for n in sorted(reported_names-true_names)]
    corrected=sum(s['size'] for s in app_globals)
    assert audit['diagnostic_globals_bytes']-corrected==sum(s['size'] for s in excluded)==33
    assert syms['xIsrStack']['size']==audit['isr_stack_static_reserved_bytes']==2096
    dis=(b/'firmware.disassembly.txt').read_text()
    matched=0
    for m in re.finditer(r'^\s*([0-9a-f]+):\s+([0-9a-f]{4,8})\s+\S',dis,re.M):
        word=m[2];assert len(word) in (4,8)
        data=int(word,16).to_bytes(len(word)//2,'little')
        assert elf.read(int(m[1],16),len(data))==data, 'Disassembly bytes differ from ELF'
        matched+=1
    active=block(dis,'ram_active_pass');probe=block(dis,'ram_stack_probe');control=block(dis,'ram_control_task(void*)');loop=block(dis,'loop()');report=block(dis,'ram_c3_emit_result()')
    assert '<hw500_predict_loaded>' in active
    assert re.search(r'\bli\s+\w+,500\b',active)
    assert 'ram_mismatches>' in active and 'ram_sink>' in active
    assert re.search(r'addi\s+sp,sp,-800\b',probe) and re.search(r'\bli\s+\w+,768\b',probe)
    assert '\tsb\t' in probe and '\tlbu\t' in probe
    assert ordered(control,['<uxTaskGetStackHighWaterMark>','<ram_stack_probe>','<uxTaskGetStackHighWaterMark>','<ram_control_task_done>'])
    baseline_call=loop.index('<ram_active_pass>')
    report_pos=loop.index('<ram_c3_emit_result()>',baseline_call)
    baseline_region=loop[baseline_call:report_pos]
    assert '<uxTaskGetStackHighWaterMark>' in baseline_region and '<heap_caps_get_info>' in baseline_region
    assert '<ram_baseline_correct>' in baseline_region and '<ram_baseline_ram_valid>' in baseline_region
    assert '<ram_active_pass>' not in report and '<heap_caps_get_info>' not in report and '<uxTaskGetStackHighWaterMark>' not in report
    assert re.search(r'\bli\s+\w+,3\b',loop[:baseline_call][-600:])
    hp=loop[loop.index('<heap_caps_malloc>')-350:]
    assert re.search(r'\bli\s+a0,512\b',hp[:500])
    assert ordered(hp,['<heap_caps_malloc>','\tsb\t','<heap_caps_get_free_size>','<heap_caps_free>','<heap_caps_get_free_size>'])
    cohort=[s for s in elf.symbols if s['type']==1 and re.match(r'_ZL\d+HC_',s['name'])]
    for s in cohort:assert s['section']=='.flash.rodata'
    model_objects=[s for s in elf.symbols if s['type']==1 and re.match(r'_ZL\d+(KC_|KLUT_|KML_|MLP16_|DT5_)',s['name'])]
    assert model_objects, 'No linked model arrays found'
    for s in model_objects:assert s['section']=='.flash.rodata'
    su=(b/'stack_usage/src/main.cpp.su').read_text()
    frames={}
    for line in su.splitlines():
        parts=line.split('\t')
        if len(parts)==3:frames[parts[0].split(':',3)[-1]]={'bytes':int(parts[1]),'kind':parts[2]}
    assert frames['void ram_stack_probe()']['bytes']==800
    return {'model':model,'elf_sha256':hashlib.sha256(elf.data).hexdigest(),'firmware_bin_sha256':hashlib.sha256((b/'firmware.bin').read_bytes()).hexdigest(),'elf_machine':'RISC-V 32-bit little-endian','static_dram_data_bss_bytes':static,'iram_text_and_alignment_bytes':iram,'isr_stack_reserved_bytes_already_in_bss':2096,'reported_diagnostic_globals_bytes':audit['diagnostic_globals_bytes'],'corrected_application_ram_globals_bytes':corrected,'excluded_non_application_objects':excluded,'diagnostic_application_globals':app_globals,'nonoverlapping_application_globals':True,'cohort_objects':cohort,'model_objects':model_objects,'cohort_and_model_placement':'flash.rodata','disassembly_instruction_words_byte_matched_to_elf':matched,'linked_evidence':{'500_row_loop_calls_prediction':True,'three_pass_loop_present':True,'baseline_snapshots_before_report':True,'report_does_not_resample_or_run_inference':True,'positive_stack_probe_frame_bytes':800,'positive_stack_payload_bytes':768,'positive_heap_alloc_fill_snapshot_free_snapshot_present':True,'positive_heap_payload_bytes':512},'compiler_individual_stack_frames_not_callgraph_peak':frames,'esp_image':esp_bin(b/'firmware.bin',elf),'status':'PASS_WITH_REPORTING_CORRECTION'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise SystemExit('Refusing existing output directory')
    args.output.mkdir(parents=True)
    results=[check_model(args.session,m) for m in ['coeff','lut','mlp','kanml','dt5']]
    data={'schema':'kanids-c3-ram142-independent-binary-review-v1','input_session':args.session.name,'results':results,'hardware_rerun':False,'rebuild':False,'status':'PASS_WITH_REPORTING_CORRECTION','limits':['Finite observed workload; neither a whole-system nor worst-case RAM proof.','C3 loop-task high-watermark is lifetime since task creation and includes earlier setup/INFO/command paths. Equal before/after only establishes no deeper observed watermark during measured workload.','Per-region historic minimum-free heap values are not a simultaneous global heap peak; pass-boundary samples can miss intra-pass transients.','Compiler .su entries are individual function frames, not task peak or whole-program call-graph maxima.','DRAM static allocations include ISR stack and application globals; do not add them twice. IRAM occupies SRAM through a separate mapping and dummy section must not be added again.','Saved disassembly instruction words were checked against actual ELF bytes; semantic review is bounded and does not constitute a formal binary proof.','Diagnostic metadata incorrectly included 33 SDK bytes via a substring match; corrected by exact application symbol prefix with ELF object type and DRAM section validation. The correction does not alter firmware or observations.']}
    (args.output/'C3_BINARY_REVIEW.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':data['status'],'models':len(results),'disassembly_words_verified':sum(r['disassembly_instruction_words_byte_matched_to_elf'] for r in results),'corrected_ram_globals':{r['model']:r['corrected_application_ram_globals_bytes'] for r in results}},indent=2))
if __name__=='__main__':main()
