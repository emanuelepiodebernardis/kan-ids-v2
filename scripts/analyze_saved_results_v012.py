#!/usr/bin/env python3
"""Recompute UNSW AUROC and validation-ratio selection from saved CSVs.

Pure stdlib; no fitting, prediction, threshold optimization or polarity flips.
Means and sample SD use the saved seed values. These are repeated fits on
fixed domains, not independent domain-level replications.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'evidence/review_v012'


def read(path):
    return list(csv.DictReader(path.open(encoding='utf-8', newline='')))


def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[]; unsw=[]
    for space,filename in [('rich','joint_training_runs_ratio5_cat.csv'),
                           ('reduced','joint_training_runs_ratio5_ridotto_cat.csv')]:
        path=REPO/'results'/filename; sources.append(path)
        groups=defaultdict(list)
        for row in read(path):
            if row['dst']=='unsw': groups[row['model']].append(row)
        assert len(groups)==6
        for model, rows in sorted(groups.items()):
            scores=[float(r['roc_auc']) for r in rows]
            assert len(scores)==10 and len({r['seed'] for r in rows})==10
            unsw.append(dict(space=space,model=model,n=len(scores),
                roc_auc_mean=mean(scores),roc_auc_sample_sd=stdev(scores),
                below_0_5=sum(x<0.5 for x in scores),
                equal_0_5=sum(x==0.5 for x in scores),
                min_roc_auc=min(scores),max_roc_auc=max(scores)))
    write_csv('saved_results_unsw_auc.csv',unsw)
    path=REPO/'results/joint_ratio_selection_runs.csv'; sources.append(path)
    groups=defaultdict(list); per_ratio=defaultdict(list)
    rows=read(path)
    for row in rows:
        assert row['split']=='validation'
        key=int(row['seed']),float(row['ratio'])
        groups[key].append(float(row['balanced_accuracy']))
        per_ratio[key[1]].append(float(row['balanced_accuracy']))
    assert len(rows)==600
    ratios=sorted(per_ratio); seeds=sorted({s for s,r in groups})
    ratio_rows=[]
    for ratio in ratios:
        values=per_ratio[ratio]
        ratio_rows.append(dict(ratio=ratio,n=len(values),mean_ba=mean(values),
            historical_mean_ba_rounded_4=round(mean(values),4),sample_sd=stdev(values)))
    historical=max(ratios,key=lambda r:round(mean(per_ratio[r]),4))
    exact=max(ratios,key=lambda r:mean(per_ratio[r]))
    winners=[]
    for seed in seeds:
        scores={ratio:mean(groups[seed,ratio]) for ratio in ratios}
        assert all(len(groups[seed,r])==12 for r in ratios)
        best=max(scores.values()); tied=[r for r in ratios if scores[r]==best]
        winner=min(tied) # pandas idxmax picks first sorted ratio if tied.
        winners.append(dict(seed=seed,winner_ratio=winner,winner_mean_ba=best,
            ratio5_mean_ba=scores[5.0],ratio5_gap_to_best=best-scores[5.0],
            ties=';'.join(str(r) for r in tied),
            **{f'mean_ba_ratio{int(r)}':scores[r] for r in ratios}))
    write_csv('saved_results_ratio_means.csv',ratio_rows)
    write_csv('saved_results_ratio_seed_winners.csv',winners)
    summary={'status':'PASS','scope':'fresh reanalysis of saved metrics; no new fits',
        'dispersion':'sample SD across saved training seeds; not domain-level uncertainty',
        'sources':[dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(read(p))) for p in sources],
        'unsw':{'groups':unsw,'counts_by_space':{
            space:{'n':sum(r['n'] for r in unsw if r['space']==space),
                   'below_0_5':sum(r['below_0_5'] for r in unsw if r['space']==space)}
            for space in ['rich','reduced']},
            'interpretation':'AUROC<0.5 describes inverted ranking with the saved attack-score polarity; threshold changes alone do not fix ranking, and no target-tuned score flip was evaluated.'},
        'ratio_selection':{'rule':'mean validation BA across six models and two domains; historical selection rounds global means to four decimals before idxmax; per-seed means are unrounded and ties choose smallest ratio',
            'historical_selected_ratio':historical,'unrounded_selected_ratio':exact,
            'ratio5_seed_wins':sum(r['winner_ratio']==5 for r in winners),'seeds':len(seeds),
            'seed_winners':winners,'global_means':ratio_rows}}
    (OUT/'saved_results_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8', newline="\n")
    print(json.dumps({'counts':summary['unsw']['counts_by_space'],
                     'ratio5_seed_wins':summary['ratio_selection']['ratio5_seed_wins'],
                     'selected_ratio':historical}))

if __name__=='__main__': main()
