"""Export neutral-named reproduction and anonymous numerical evidence packages.

Executed source bytes are preserved. A path ledger restores their original
relative layout for replay; no recorded source digest or role salt is rewritten.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, subprocess, zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SETS = {'initial': HERE, 'diagnostic_a': HERE/'review2',
        'diagnostic_b': HERE/'review3', 'diagnostic_c': HERE/'calibration_audit'}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tracked():
    return [REPO/x for x in subprocess.check_output(
        ['git','ls-files'], cwd=REPO, text=True).splitlines()]

def release_destination(rel):
    """Return the neutral export path for a tracked repository path.

    Compare path components instead of their string representation so the
    export behaves identically with POSIX and Windows separators.
    """
    parts = tuple(rel.parts)
    review_prefix = ('adattamento-drift', 'review_extension')
    temporal_prefix = ('adattamento-drift', 'ton_temporal')
    if parts[:len(review_prefix)] == review_prefix:
        r = Path(*parts[len(review_prefix):])
        if not r.parts:
            return None
        if rel.suffix == '.md' and rel.name != 'PROTOCOL.md':
            return None
        if rel.name in {
            'ARTIFACT_MANIFEST.json', 'qa_summary.json', 'export_release.py'
        }:
            return None
        if (rel.suffix in {'.tex', '.bib', '.bbl', '.pdf', '.png', '.txt'}
                and 'results' not in r.parts and 'logs' not in r.parts):
            return None
        if r.parts[0] == 'archive':
            return None
        key = 'extension'
        if r.parts[0] in {'review2', 'review3', 'calibration_audit'}:
            key = {
                'review2': 'diagnostic_a',
                'review3': 'diagnostic_b',
                'calibration_audit': 'diagnostic_c',
            }[r.parts[0]]
            r = Path(*r.parts[1:])
        return Path('source') / key / r
    if parts[:len(temporal_prefix)] == temporal_prefix:
        r = Path(*parts[len(temporal_prefix):])
        if (rel.suffix not in {'.py', '.json', '.csv', '.npz', '.pkl'}
                and rel.name != 'rapporto.md'):
            return None
        return Path('source/temporal') / r
    return None

def zipdir(root, output):
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p, root.name+'/'+p.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as z: assert z.testzip() is None

RESTORE = '''from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
m=json.loads((root/'SOURCE_LEDGER.json').read_text())
if a.out.exists() and any(a.out.iterdir()):raise SystemExit('Use an empty output directory')
for src,item in m['files'].items():
    data=(root/src).read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['sha256'],src
    dest=a.out/item['execution_path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
e=a.out/'adattamento-drift/review_extension'
for name in ['manuscript/figures','manuscript/tables']:(e/name).mkdir(parents=True,exist_ok=True)
hashes={str(p.relative_to(e)):hashlib.sha256(p.read_bytes()).hexdigest() for p in e.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json'}
(e/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'scope':'filtered reproduction export; executed source bytes unchanged','sha256':hashes},indent=2)+'\\n')
subprocess.run(['git','init','-q',str(a.out)],check=True)
subprocess.run(['git','add','.'],cwd=a.out,check=True)
subprocess.run(['git','-c','user.name=Reproduction snapshot','-c','user.email=reproduction@example.invalid','commit','-q','-m','Restore hash-verified execution snapshot; local provenance only'],cwd=a.out,check=True)
print('Restored',len(m['files']),'hash-verified files to',a.out.resolve())
'''

CHECK = '''from pathlib import Path
import csv,gzip,hashlib,json,math
root=Path(__file__).resolve().parent
m=json.loads((root/'SHA256.json').read_text())
for name,h in m.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
checks=0;runs=0
for p in sorted((root/'reports').rglob('*.json.gz')):
    r=json.load(gzip.open(p,'rt'));runs+=1
    for scope,policies in r['summary'].items():
        for policy,metrics in policies.items():
            counts=[sum(b['cm'][scope][policy][i] for b in r['per_block']) for i in range(4)]
            for key,v in zip(['tn','fp','fn','tp'],counts):
                assert metrics[key]==v,(p,scope,policy,key);checks+=1
            tn,fp,fn,tp=counts
            recall=tp/(tp+fn) if tp+fn else None
            fpr=fp/(tn+fp) if tn+fp else None
            for key,v in [('recall',recall),('fpr',fpr),('ba',(recall+1-fpr)/2 if recall is not None and fpr is not None else None)]:
                if v is not None:assert math.isclose(metrics[key],v,rel_tol=1e-12,abs_tol=1e-12),(p,key);checks+=1
assert runs==195,runs
print('Verified',len(m),'file hashes,',runs,'recorded cells and',checks,'count/rate equalities. Thirty initial cells are repeated controls, not independent replications.')
'''

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    full=out/'Paper2_Reproduction';anon=out/'Numerical_Evidence'
    for d in [full,anon]:
        if d.exists():shutil.rmtree(d)
        d.mkdir()
    ledger={}
    for p in tracked():
        rel=p.relative_to(REPO)
        if not p.is_file():continue
        dest=release_destination(rel)
        if dest is None:continue
        target=full/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        ledger[dest.as_posix()]={'execution_path':rel.as_posix(),'sha256':digest(p)}
    required_sources = {
        'source/extension/experiment.py',
        'source/diagnostic_a/runner.py',
        'source/diagnostic_b/runner.py',
        'source/diagnostic_c/runner.py',
        'source/temporal/replay.py',
    }
    missing_sources = sorted(required_sources - set(ledger))
    if missing_sources:
        raise RuntimeError(f'reproduction export is incomplete: {missing_sources}')
    (full/'SOURCE_LEDGER.json').write_text(json.dumps({'files':ledger},indent=2)+'\n')
    (full/'restore_execution.py').write_text(RESTORE)
    for label,src in SETS.items():
        for p in sorted((src/'analysis').glob('*.csv')):
            for base in [full,anon]:
                target=base/'analysis'/label/p.name;target.parent.mkdir(parents=True,exist_ok=True)
                # Literal salt bytes are retained only in the full execution provenance.
                text=p.read_text().replace('review2-salt-1','salt1').replace('review2-salt-2','salt2')
                target.write_text(text)
        for p in sorted((src/'results').glob('*.json.gz')):
            r=json.load(gzip.open(p,'rt'))
            keep=['dataset','family','seed','condition','budget','delay','block_size','summary','per_block']
            clean={k:r[k] for k in keep if k in r}
            target=anon/'reports'/label/p.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(gzip.compress(json.dumps(clean,separators=(',',':'),allow_nan=True).encode(),mtime=0))
    (anon/'verify_numerical_evidence.py').write_text(CHECK)
    (anon/'README.txt').write_text('Numerical evidence for Thresholds and Learning in Adaptive Intrusion Detection\n\nRun: python verify_numerical_evidence.py\nPython 3 standard library only; no raw data or model deserialization.\n\n195 recorded cells: initial extension 55, diagnostic A 120, B 10, C 10.\nThirty A cells repeat earlier trajectories. Seeds are conditional on one\nstream per dataset. This package checks recorded block counts and rates;\nit does not rerun predictions. Identifying execution provenance and fitted\nmodels are supplied separately to the editors. CSV files retain stable\nmachine policy keys; paper labels ALL256 and ROLE64/256 specify eligibility.\n')
    (anon/'SHA256.json').write_text(json.dumps({p.relative_to(anon).as_posix():digest(p) for p in sorted(anon.rglob('*')) if p.is_file()},indent=2)+'\n')
    shutil.copyfile(HERE/'REPRODUCTION.txt',full/'README.txt')
    shutil.copyfile(REPO/'LICENSE',full/'CODE_LICENSE.txt')
    (full/'SHA256.json').write_text(json.dumps({p.relative_to(full).as_posix():digest(p) for p in sorted(full.rglob('*')) if p.is_file()},indent=2)+'\n')
    zipdir(full,out/'Paper2_Reproduction.zip');zipdir(anon,out/'Numerical_Evidence.zip')
    print(json.dumps({'source_files':len(ledger),'reproduction_zip':str(out/'Paper2_Reproduction.zip'),'anonymous_zip':str(out/'Numerical_Evidence.zip')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
