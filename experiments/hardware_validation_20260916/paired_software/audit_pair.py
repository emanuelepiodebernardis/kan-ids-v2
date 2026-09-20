"""Read-only audit of supplied pair-study artifacts. No model loading or training.
Requires Python >=3.9 and NumPy. Run alongside verify_saved_study.py.
"""
import argparse, csv, gzip, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
from verify_saved_study import metric


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage1',type=Path,required=True)
    ap.add_argument('--stage2',type=Path,required=True)
    ap.add_argument('--kit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); out=a.out.resolve()
    if out.exists() or any(out.is_relative_to(p.resolve()) for p in (a.stage1,a.stage2,a.kit)):
        ap.error('Output must be new and outside inputs')
    errors=[]; checks=0; manifest_counts={}
    def check(k,v):
        nonlocal checks
        checks+=1
        if not v: errors.append(k)
    for label,root,name in [('stage1',a.stage1,'RUN_MANIFEST.json'),('stage2',a.stage2,'RUN_MANIFEST.json'),('kit',a.kit,'KIT_MANIFEST.json')]:
        manifest=read(root/name)['files']; manifest_counts[label]=len(manifest)
        for f in manifest:
            p=(root/f['path']).resolve()
            check(label+':safe:'+f['path'],p.is_relative_to(root.resolve()) and not p.is_symlink())
            check(label+':hash:'+f['path'],p.is_file() and sha(p)==f['sha256'])
            check(label+':bytes:'+f['path'],p.stat().st_size==f['bytes'])
    source=a.kit/'data/train_test_network.csv'
    for label,root in [('stage1',a.stage1),('stage2',a.stage2)]:
        protocol=read(root/'PROTOCOL.json')
        check(label+':source-hash',sha(source)==protocol['source']['sha256'])
    stage1p=read(a.stage1/'PROTOCOL.json'); stage2p=read(a.stage2/'PROTOCOL.json')
    for k in ['source','split','new_preprocessing','models','new_unknown_category_policy','planned_model_seeds']:
        check('unchanged-protocol:'+k,stage1p[k]==stage2p[k])
    check('identical-split',sha(a.stage1/'audit/split_assignments.csv.gz')==sha(a.stage2/'audit/split_assignments.csv.gz'))
    source_files=0
    for f in (a.stage2/'source').rglob('*'):
        if not f.is_file() or f.name=='KIT_MANIFEST.json': continue
        rel=f.relative_to(a.stage2/'source')
        other=a.kit/('reference' if rel.parts[0]=='kanids' else '')/rel
        check('stage2-source-kit:'+str(rel),other.is_file() and sha(f)==sha(other)); source_files+=1
    with gzip.open(a.stage2/'audit/split_assignments.csv.gz','rt',encoding='utf-8',newline='') as h: assignments=list(csv.DictReader(h))
    counts={s:Counter() for s in ['train','validation','test']}; pairs={s:set() for s in counts}; hosts={s:set() for s in counts}; full={s:set() for s in counts}; ids={s:[] for s in counts}
    labels=[]; types=[]; invalid=[]; count=0
    with source.open(encoding='utf-8-sig',newline='') as h:
        r=csv.DictReader(h); columns=r.fieldnames
        for i,row in enumerate(r):
            count+=1; key=json.dumps(sorted([row['src_ip'],row['dst_ip']]),ensure_ascii=False,separators=(',',':'))
            digest=hashlib.sha256((stage2p['split']['salt']+'\0'+key).encode()).digest(); bucket=int.from_bytes(digest[:8],'big')%100
            s='train' if bucket<60 else 'validation' if bucket<80 else 'test'
            if i>=len(assignments) or assignments[i]!={'row_id':str(i),'split':s,'group_sha256':digest.hex()}: invalid.append(i)
            counts[s]['rows']+=1; counts[s][row['type']]+=1; pairs[s].add(key); hosts[s].update([row['src_ip'],row['dst_ip']]); ids[s].append(i)
            full[s].add(tuple(row[c] for c in columns)); labels.append(int(row['label'])); types.append(row['type'])
    check('all-assignments-recomputed',not invalid and count==len(assignments)==211043)
    audit=read(a.stage2/'audit/DATA_AUDIT.json')
    for s in counts:
        expected=audit['splits'][s]
        check(s+':rows',counts[s]['rows']==expected['rows'])
        check(s+':types',{k:v for k,v in counts[s].items() if k!='rows'}==expected['types'])
        check(s+':pairs',len(pairs[s])==expected['pair_groups'])
        check(s+':rowhash',hashlib.sha256(np.array(ids[s],dtype='<i8').tobytes()).hexdigest()==expected['row_ids_sha256'])
    overlap={}
    for x,y in [('train','validation'),('train','test'),('validation','test')]:
        d={'shared_pair_groups':len(pairs[x]&pairs[y]),'shared_full_row_keys':len(full[x]&full[y]),'shared_hosts':len(hosts[x]&hosts[y])}
        overlap[x+'_to_'+y]=d
        for k,v in d.items(): check(x+':'+y+':'+k,v==audit['overlap'][x+'_to_'+y][k])
    with gzip.open(a.stage1/'pilot/validation_scores.csv.gz','rt',encoding='utf-8',newline='') as h: rows=list(csv.DictReader(h))
    pilot=read(a.stage1/'pilot/validation_metrics.json')
    check('stage1-validation-row-order',[int(r['row_id']) for r in rows]==ids['validation'])
    y=np.array([int(r['y_true']) for r in rows]); check('stage1-source-labels',list(y)==[labels[i] for i in ids['validation']])
    for m in ['kan','dt5','mlp16','gam']:
        p=np.array([float(r[m+'_p_attack']) for r in rows]); pred=np.array([int(r[m+'_prediction']) for r in rows])
        check('stage1-decisions:'+m,bool(np.array_equal(pred,p>=.5)))
        actual=metric(y,p)
        for k,v in pilot['models'][m].items(): check('stage1-metric:'+m+':'+k,math.isclose(actual[k],v,rel_tol=0,abs_tol=1e-12))
    identical=[]; different=[]
    for f in (a.stage1/'pilot').iterdir():
        q=a.stage2/'study/fits/seed_20260916'/f.name
        if f.is_file() and q.is_file() and f.suffix in ('.joblib','.npz'):
            (identical if sha(f)==sha(q) else different).append(f.name)
    check('stage1-stage2-preprocessor-metadata',read(a.stage1/'pilot/shared_preprocessor.json')==read(a.stage2/'study/fits/seed_20260916/shared_preprocessor.json'))
    result={'status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,'manifest_files_checked':manifest_counts,'source_snapshot_files_matched_to_kit':source_files,'source_csv_sha256':sha(source),'source_rows':count,'source_columns':len(columns),'split_rows':{s:counts[s]['rows'] for s in counts},'pair_groups':{s:len(pairs[s]) for s in counts},'overlap_recomputed':overlap,'stage1_validation_predictions_checked':len(rows)*4,'stage1_stage2_first_seed_identical_checkpoints':identical,'stage1_stage2_first_seed_byte_different_checkpoints':different,'preprocessor_json_identical':True,'method':'All source-row pair assignments independently recomputed; decoded 44-column tuples compared directly for duplicates; manifests/source snapshots verified; stage1 validation metrics recomputed from saved probabilities. No model loading or training.','limitations':['Run manifests attest supplied artifact consistency, not independently witnessed execution.','Model-input overlap masks are checked separately for stored-score consistency; this audit does not load fitted preprocessors to recompute transformed inputs.']}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2)); return bool(errors)
if __name__=='__main__': raise SystemExit(main())
