"""Execute every frozen protocol cell, with resumable per-cell outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--families',nargs='+',default=['spline','lr','arf'])
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);(a.out/'logs').mkdir(exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    jobs=[(dataset,family,seed) for family in a.families for seed in a.seeds for dataset in ('ton','unsw')]
    def job(spec):
        dataset,family,seed=spec;prefix=f'{dataset}_{family}_{seed}'
        base=[sys.executable,str(HERE/'experiment.py')]
        common=['--data',str(a.data_root/dataset),'--cache',str(a.cache_root/dataset),
                '--family',family,'--seed',str(seed)]
        log=a.out/'logs'/(prefix+'.log')
        with log.open('a') as f:
            subprocess.run(base+['prepare']+common,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            conditions=[('main',1,.01,False)]
            if dataset=='ton' and family=='spline':
                conditions += [('legacy',1,.01,True),('delay0',0,.01,False),
                               ('delay5',5,.01,False),('delay20',20,.01,False),
                               ('budget001',1,.001,False)]
            for label,delay,budget,legacy in conditions:
                out=a.out/(prefix+'_'+label+'.json.gz')
                if out.exists():continue
                cmd=base+['run']+common+['--out',str(out),'--delay',str(delay),'--budget',str(budget)]
                if legacy:cmd+=['--legacy']
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('completed',prefix,flush=True)
        return prefix
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(job,j) for j in jobs]
        failures=[]
        for j,f in zip(jobs,futures):
            try:f.result()
            except Exception as exc:failures.append({'job':j,'error':str(exc)});print('FAILED',j,str(exc),flush=True)
    (a.out/('campaign_status_'+'_'.join(a.families)+'.json')).write_text(json.dumps({'jobs':jobs,'failures':failures},indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
