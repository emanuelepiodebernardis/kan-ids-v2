"""Read-only HW500 receipt audit; output must be a fresh separate directory.
Use the unmodified v0.13.3 kit as --kit. Requires that kit's pinned numpy/scipy.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

p=argparse.ArgumentParser()
p.add_argument('--session',required=True,type=Path)
p.add_argument('--kit',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
sys.dont_write_bytecode=True
session=a.session.resolve();kit=a.kit.resolve();out=a.out.resolve()
if out.is_relative_to(session):raise ValueError('Output must be outside original evidence')
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(kit))
from run_suite import verify_packet,verify_sources,digest,read_json
from analyze_continuous import analyze_session_trace
verify_sources(kit);verify_sources(session/'sources')
assert digest(kit/'KIT_MANIFEST.json')==digest(session/'sources/KIT_MANIFEST.json')
for entry in read_json(kit/'KIT_MANIFEST.json')['files']:
 assert digest(kit/entry['path'])==digest(session/'sources'/entry['path'])
packets=sorted(session.rglob('PACKET_MANIFEST.json'))
for packet in packets:verify_packet(packet.parent)
s=read_json(session/'SUITE_RECORD.json');records=[]
for e in s['attempts']:
 r=session/e['relative_path']
 assert digest(r/'RUN_RECORD.json')==e['record_file']
 assert digest(r/'SOFTWARE_RECORD.json')==e['software_record_file']
 assert digest(r/'SOFTWARE_MANIFEST.json')==e['software_manifest_file']
 for item in read_json(r/'SOFTWARE_MANIFEST.json')['files']:
  assert digest(r/item['path'])=={'bytes':item['bytes'],'sha256':item['sha256']}
 records.append(read_json(r/'SOFTWARE_RECORD.json'))
records.sort(key=lambda r:r['plan_item']['index'])
raws=list((session/'captures').glob('*.cfn'));assert len(raws)==1
fresh=analyze_session_trace(raws[0],records)
old=read_json(session/s['analysis_attempts'][0]['path']/'SERIES_ANALYSIS.json')
maximum=0.
for x,y in zip(fresh['analyses'],old['analyses']):
 assert x['run_id']==y['run_id'] and x['workload']==y['workload']
 for key in ('energy_J','mean_power_W','uJ_per_inference','duration_CFN_s'):
  assert math.isclose(x['primary_energy'][key],y['primary_energy'][key],rel_tol=1e-6,abs_tol=1e-8)
 maximum=max(maximum,abs(x['primary_energy']['uJ_per_inference']-y['primary_energy']['uJ_per_inference']))
# Independently reproduce unweighted run-level means/SD, excluding pilot.
with (session/'results_model_means.csv').open(newline='',encoding='utf-8') as f: table=list(csv.DictReader(f))
for row in table:
 group=[x for x in fresh['analyses'] if x['model']==row['model'] and x['phase']=='campaign']
 assert len(group)==5
 values={'latency_us':[x['workload']['mean_replay_us_per_inference'] for x in group],
         'USB_power_W':[x['primary_energy']['mean_power_W'] for x in group],
         'USB_energy_uJ':[x['primary_energy']['uJ_per_inference'] for x in group]}
 for metric,nums in values.items():
  for suffix,result in [('mean',statistics.mean(nums)),('sample_SD',statistics.stdev(nums))]:
   assert math.isclose(result,float(row[metric+'_'+suffix]),rel_tol=1e-6,abs_tol=1e-8)
report={'status':'PASS','verified_packet_manifests':len(packets),'recomputed_runs':len(fresh['analyses']),
        'raw_CFN':digest(raws[0]),'maximum_absolute_energy_difference_uJ':maximum,
        'means_and_sample_SD_from_five_campaign_runs_per_model':'PASS',
        'scope':'Software and numerical evidence validation; does not establish absolute instrument accuracy'}
(out/'SERIES_ANALYSIS_RECOMPUTED.json').write_text(json.dumps(fresh,indent=2)+'\n',encoding='utf-8')
(out/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
