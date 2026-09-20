"""Derive publication summaries from the retained HW500 session artifacts.
This does not acquire hardware data. Pass --sessions for relocated extracted raw ZIPs.
"""
from pathlib import Path
import argparse,csv,hashlib,json,statistics
p=argparse.ArgumentParser();p.add_argument('--sessions',type=Path,default=Path(__file__).resolve().parent/'raw_sessions');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
models=['coeff','lut','mlp','kanml','dt5'];names=['Coefficient KAN','LUT KAN','MLP16','Multilayer KAN','DT5']
boards={};allruns=[]
for board,prefix in [('mega','MEGA'),('c3','C3')]:
 session=next(a.sessions.glob(prefix+'_HW500_CONTINUOUS_*'))
 summary=json.loads((session/'SUMMARY.json').read_text());rows=list(csv.DictReader((session/'results_all_runs.csv').open()))
 series=json.loads(next(session.glob('analysis_attempts/*/SERIES_ANALYSIS.json')).read_text())
 assert summary['confirmatory_runs']==25 and summary['separate_pilot_runs']==1 and not summary['retained_failed_attempts']
 assert len(rows)==26 and len(series['analyses'])==26
 for row in rows:
  for k in row.keys()-{'board','model','phase','run_id'}: row[k]=float(row[k])
  for k in ('N','full_traversals','block'): row[k]=int(row[k])
  allruns.append(row)
 means=[]
 for model in models:
  g=[r for r in rows if r['model']==model and r['phase']=='campaign']; assert len(g)==5
  meansrow={'board':board,'model':model,'technical_repeats':5,'physical_boards':1}
  for metric in ('latency_us','USB_power_W','USB_energy_uJ'):
   vals=[r[metric] for r in g]
   meansrow[metric+'_mean']=statistics.mean(vals);meansrow[metric+'_sample_SD']=statistics.stdev(vals)
  src=next(r for r in summary['mean_rows'] if r['model']==model)
  assert all(abs(meansrow[k]-src[k])<=1e-12 for k in meansrow if k.endswith(('_mean','_sample_SD')))
  meansrow.update(N_per_batch=[r['N'] for r in g],active_MCU_s=[r['active_MCU_s'] for r in g],energy_uJ_runs=[r['USB_energy_uJ'] for r in g],idle_subtracted_signed_diagnostic_uJ_mean=statistics.mean(r['incremental_uJ_diagnostic'] for r in g))
  means.append(meansrow)
 boards[board]={'session':session.name,'source_CFN_sha256':series['source_sha256'],'accepted_runs':26,'included_runs':25,'excluded_pilots':1,'failed_attempts':[],'means':means,'record_count':series['analyses'][0]['recording_integrity']['records'],'record_span_s':series['analyses'][0]['recording_integrity']['time_grid']['span_s'],'maximum_marker_residual_s':max(r['registration_max_residual_s'] for r in rows),'maximum_equal_endpoint_shift_2s_sensitivity_percent':max(r['registration_shift_sensitivity_percent'] for r in rows),'interpretation':summary['interpretation']}
 coeff,lut=means[0],means[1];boards[board]['coefficient_divided_by_LUT_latency']=coeff['latency_us_mean']/lut['latency_us_mean'];boards[board]['coefficient_divided_by_LUT_energy']=coeff['USB_energy_uJ_mean']/lut['USB_energy_uJ_mean']
report={'schema':'kanids-hw500-publication-integration-v1','source_protocol':'kanids-hw500-continuous-v1; kit v0.13.3','cohort':{'distinct_flows':500,'attack':250,'normal':250,'frozen_NPZ_sha256':'20c53b571bfaef91f5fd26b19e5960e3fa85ed48b38c78c8527245b921ebe86f'},'boundary':'Prepared Flash row -> RAM buffer -> integer prediction -> checksum; identical batch for timing and energy','energy_method':'Integral VBUS*IBUS over per-run LED-registered CFN active interval / completed inference count','time_method':'MCU active microseconds / completed inference count; not individual-request latency distribution','primary_idle_subtraction':False,'NRG_used':False,'calibrated':False,'model_identity':'Historical frozen canonical models, not newly fitted pair-disjoint Stage2 models','boards':boards,'limitations':['One physical board of each type; five technical repeats, not independent devices.','No independent meter or MCU/CFN clock calibration; no full uncertainty budget.','Nominal0.1s stored grid is not10Hz physical bandwidth.','Mean and sample SD measure within-setup repeatability only.','Marker-fit residuals and common endpoint shifts are diagnostics, not calibrated timing uncertainty.','Includes board/regulator/indicator/downstream wiring consumption; not isolated CPU or end-to-end IDS.','Negative signed idle differences on Mega retained as diagnostics; not clamped or used for primary estimates.','NRG is approximately one quarter of integrated VI; unknown cause; retained but excluded without factor4 correction.','Original C3+7.58% slowdown remains a separate protocol-dependent historical result; these results do not identify its cause.','20-flow energy pilots and kernel-only historical latency remain separate; no pooled means.']}
(a.output/'HW500_PUBLICATION_DATA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (a.output/'HW500_RUNS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allruns[0]));w.writeheader();w.writerows(allruns)
for lang in ['EN','RU']:
 caption=('Common 500-flow hardware workload: time and whole-board USB energy from the same active batch. Five technical repetitions per model and board; energy is mean $\\pm$ sample SD. The separate coefficient pilot is excluded.' if lang=='EN' else 'Общая нагрузка из 500 потоков: время и USB-энергия всей платы для одного активного блока. Для каждой модели и платы выполнено пять технических повторов; энергия представлена как среднее $\\pm$ выборочное SD. Отдельный коэффициентный пилот исключён.')
 rows=['\\begin{table*}[t]','\\centering','\\caption{'+caption+'}','\\label{tab:hw500-common}','\\begin{tabular}{lrrrr}','\\toprule',('Model' if lang=='EN' else 'Модель')+' & \\multicolumn{2}{c}{Mega 2560} & \\multicolumn{2}{c}{ESP32-C3} \\\\',(' & Time ($\\mu$s) & Energy ($\\mu$J) & Time ($\\mu$s) & Energy ($\\mu$J) \\\\' if lang=='EN' else ' & Время (мкс) & Энергия (мкДж) & Время (мкс) & Энергия (мкДж) \\\\'),'\\midrule']
 for i,(m,name) in enumerate(zip(models,names)):
  mg=boards['mega']['means'][i];c3=boards['c3']['means'][i]
  rows.append(f"{name} & {mg['latency_us_mean']:.3f} & ${mg['USB_energy_uJ_mean']:.3f}\\pm{mg['USB_energy_uJ_sample_SD']:.3f}$ & {c3['latency_us_mean']:.3f} & ${c3['USB_energy_uJ_mean']:.4f}\\pm{c3['USB_energy_uJ_sample_SD']:.4f}$ \\\\")
 rows.extend(['\\bottomrule','\\end{tabular}','\\end{table*}'])
 (a.output/f'HW500_TABLE_{lang}.tex').write_text('\n'.join(rows)+'\n')
print('Wrote tables and verified all10mean/SD rows against retained summaries')
