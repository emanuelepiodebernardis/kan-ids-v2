#!/usr/bin/env python3
"""Verify v0.12 manuscript tables against preserved numerical evidence and photos."""
import csv,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def flat(s):return re.sub(r'\s+',' ',s).strip()
def main():
 tex=[R/'main.tex',*sorted((R/'inputs').glob('*.tex'))]
 current='\n'.join(p.read_text() for p in tex)
 energy=(R/'inputs/energy_full_precision.tex').read_text()
 means=json.loads((R/'evidence/energy_pilot_results.json').read_text())['means']
 names={'KAN coefficients':'coeff','KAN sampled LUT':'lut','Multilayer KAN':'kanml','MLP16':'mlp','DT5':'dt5'}
 n=0
 for name,cells in re.findall(r'^(KAN coefficients|KAN sampled LUT|Multilayer KAN|MLP16|DT5) & (.+?)\\\\$',energy,re.M):
  row={x['board']:x for x in means if x['model']==names[name]};m,c=row['Mega 2560'],row['ESP32-C3']
  expected=[f"{m['mean_call_us']:.3f}",f"{m['mean_power_W']*1000:.2f}",f"{m['energy_uJ_per_call']:.4f}",f"{c['mean_call_us']:.4f}",f"{c['mean_power_W']*1000:.2f}",f"{c['energy_uJ_per_call']:.4f}"]
  assert [v.strip() for v in cells.split('&')]==expected,name;n+=6
 assert n==30
 rows=list(csv.DictReader((R/'evidence/subgroup_metrics.csv').open()))
 subgroup=(R/'inputs/duplicate_sensitivity.tex').read_text()
 labels={'all':'All','exact_duplicate':'Exact training match','non_overlapping':'No exact training match'}
 for key,label in labels.items():
  row=next(x for x in rows if x['representation']=='coefficient' and x['subgroup']==key)
  values=[f"{int(row[k]):,}" for k in ['n','normal','attack','TN','FP','FN','TP']]
  values+=[f"{1-float(row['TNR']):.4f}".removeprefix('0'),f"{float(row['TPR']):.4f}".removeprefix('0'),f"{float(row['BA']):.4f}".removeprefix('0')+' / '+f"{float(row['F1']):.4f}".removeprefix('0')]
  assert label+' & '+' & '.join(values) in subgroup,label
  for rep in ['lut1025','lut513']:
   other=next(x for x in rows if x['representation']==rep and x['subgroup']==key)
   assert all(other[k]==row[k] for k in ['n','normal','attack','TN','FP','FN','TP','F1','BA','TNR','TPR'])
 cert=(R/'inputs/duplicate_certificates.tex').read_text()
 certlabels={'all':'All','exact_duplicate':'Exact match','non_overlapping':'No exact match'}
 for row in csv.DictReader((R/'evidence/subgroup_certificates.csv').open()):
  n=int(row['n']);u=int(row['uncertified_signed_lut_guard']);assert int(row['coefficient_lut_disagreements'])==0
  assert f"{certlabels[row['subgroup']]} & {row['L']} & {n:,} & {u} & {n-u:,}" in cert
 auroc=(R/'inputs/unsw_auroc.tex').read_text()
 am={'KAN(cat,1L)':'Additive KAN','KAN(cat,ML)':'Multilayer KAN','MLP(16)':'MLP16','DecisionTree(d=5)':'DT5','LightGBM':'LightGBM','XGBoost':'XGBoost'}
 for row in csv.DictReader((R/'evidence/saved_results_unsw_auc.csv').open()):
  if row['space']=='rich':assert f"{am[row['model']]} & ${float(row['roc_auc_mean']):.4f}\\pm{float(row['roc_auc_sample_sd']):.4f}$ & {row['below_0_5']}/10" in auroc
 photos=json.loads((R/'figures/PHOTO_PROVENANCE.json').read_text())
 for item in photos:
  b=(R/'figures'/item['paper_asset']).read_bytes();assert len(b)==item['bytes'];assert hashlib.sha256(b).hexdigest()==item['sha256']
 metadata=json.loads((R/'AUTHOR_METADATA.json').read_text());authors=['Oleksandr Kuznetsov','Emanuele Pio De Bernardis','Emanuele Frontoni']
 assert [x['name'] for x in metadata['authors']]==authors
 maintex=(R/'main.tex').read_text();assert 'pdfauthor={'+'; '.join(authors)+'}' in maintex
 assert 'p=0.083' not in current and '0.99617' in current
 assert 'no runtime\nguard or fallback has been added' in current
 assert '7.58\\%' in maintex and 'unresolved' in maintex
 baseline=R.parents[1]/'paper1_hardware_integration_20260915/en'
 retained=0
 if baseline.exists():
  # Compare every scientific table row independent of position, caption or row order.
  whole=flat(current)
  for p in [baseline/'main.tex',*sorted((baseline/'inputs').glob('*.tex'))]:
   for body in re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',p.read_text(),re.S):
    for line in body.splitlines():
     if '&' in line and re.search(r'\d',line) and not any(t in line for t in ['multicolumn','cmidrule','Implementation','Model','Parameters','Normal:attack']):
      assert flat(line) in whole,(p.name,line);retained+=1
 return {'status':'PASS','version':'0.12.0','energy_full_precision_cells':30,'subgroups':3,'certificate_rows':6,'rich_auc_rows':6,'legacy_numeric_table_rows_retained':retained,'photo_bytes_match':True,'author_order':authors,'scope':'Table consistency and asset integrity; no new fitting, physical runs or absolute calibration'}
if __name__=='__main__':print(json.dumps(main(),indent=2))
