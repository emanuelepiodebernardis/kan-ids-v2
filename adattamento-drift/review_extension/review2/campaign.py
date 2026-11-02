"""Run every second-review cell in its own process; preserve complete logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from runner import CONFIGS
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--families',nargs='+',default=['spline','lr','arf']);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--conditions',nargs='+',choices=list(CONFIGS))
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);(a.out/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    jobs=[]
    for family in a.families:
        conditions=list(CONFIGS)[:8] if family=='spline' else ['main'] if family=='lr' else ['main','shuffled','fit_only']
        if a.conditions:conditions=[c for c in conditions if c in a.conditions]
        for condition in conditions:
            for seed in a.seeds:
                for dataset in ('ton','unsw'):jobs.append((dataset,family,seed,condition))
    def job(j):
        ds,family,seed,condition=j;name=f'{ds}_{family}_{seed}_{condition}'
        target=a.out/(name+'.json.gz')
        if target.exists():return name
        with (a.out/'logs'/(name+'.log')).open('a') as f:
            subprocess.run([sys.executable,str(HERE/'runner.py'),'--data',str(a.data_root/ds),
                '--cache-root',str(a.cache_root),'--out',str(target),'--family',family,
                '--seed',str(seed),'--condition',condition],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        return name
    status={'jobs':jobs,'completed':[],'failures':[]}
    statusfile=a.out/('status_'+'_'.join(a.families)+'.json')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(job,j):j for j in jobs}
        for f in as_completed(pending):
            try:name=f.result();status['completed'].append(name);print('DONE',name,flush=True)
            except Exception as ex:status['failures'].append({'job':pending[f],'error':str(ex)});print('FAILED',pending[f],flush=True)
            statusfile.write_text(json.dumps(status,indent=2)+'\n')
    if status['failures']:raise SystemExit(1)

if __name__=='__main__':main()
