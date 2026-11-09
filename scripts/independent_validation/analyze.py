"""Named metrics, paired integrity gate and prespecified conditional time intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from common import (PACKAGE, RESULTS, verify_package, verify_run, manifest, pilot, read, sha,
                    require, utc, load_cache, metric_values, relative_improvement, atomic_json,
                    atomic_npz, write_csv)

METRICS=('mse_std','mae_std','rmse_std','r2','mae_kw','rmse_kw')
PAIRS=(('residual','history'),('direct','history'),('residual','direct'))


def block_weights(length,draws=10000,days=365,seed=20260927):
    """Calendar-day circular moving blocks; empty dates remain in the calendar."""
    require(0<length<=days and draws>0,'Invalid block specification')
    rng=np.random.RandomState(seed)
    starts=rng.randint(0,days,size=(draws,int(np.ceil(days/length))))
    indices=((starts[:,:,None]+np.arange(length)[None,None,:])%days).reshape(draws,-1)[:,:days]
    weights=np.zeros((draws,days),dtype=np.int16)
    for i in range(draws):
        weights[i]=np.bincount(indices[i],minlength=days)
    require(np.all(weights.sum(axis=1)==days),'Invalid resampling weights')
    return weights


def daily_loss(error,origin_ns):
    error=np.asarray(error,dtype=np.float64)
    require(error.ndim==2 and len(error)==len(origin_ns),'Invalid daily loss shape')
    day=(np.asarray(origin_ns,dtype='datetime64[ns]').astype('datetime64[D]')-
         np.datetime64('2022-01-01','D')).astype(int)
    require(((day>=0)&(day<365)).all(),'Origin outside fixed 2022 calendar')
    squared=np.bincount(day,weights=np.sum(error**2,axis=1),minlength=365)
    absolute=np.bincount(day,weights=np.sum(np.abs(error),axis=1),minlength=365)
    counts=np.bincount(day,weights=np.full(len(day),error.shape[1]),minlength=365)
    return squared,absolute,counts


def resampled_loss(sums,counts,weights):
    denominator=weights@counts
    # Do not drop inconvenient replicates or silently resample until nonempty.
    require((denominator>0).all(),'A bootstrap draw has no data; explicit protocol review required')
    return (weights@sums)/denominator


def check_predictions(row,package,results,lock_hash):
    out=results/'runs'/row['run_id']
    summary=verify_run(out,row,lock_hash)
    cache=load_cache(package,row,'test')
    with np.load(out/'test_predictions.npz',allow_pickle=False) as z:
        predictions={k:z[k] for k in z.files}
    for a,b in (('true_std','target_scaled'),('true_kw','target_kw'),('raw_kw','target_raw_kw'),
                ('origin_ns','origin_ns'),('target_ns','target_ns')):
        require(np.array_equal(predictions[a],cache[b]),'Unpaired truth or time axis: '+row['run_id'])
    scaler=read(package/'train_only_scalers.json')[row['site']]['power']
    scale,mean=scaler['scale'][0],scaler['mean'][0]
    require(np.array_equal(predictions['pred_kw'],predictions['pred_std']*scale+mean),'Prediction unit mismatch')
    metrics=metric_values(predictions['pred_std'],predictions['true_std'],scale)
    recorded=read(out/'metrics.json')
    for k,v in metrics.items():
        require(recorded[k]==v,'Named metric mismatch: '+row['run_id']+'/'+k)
    epochs=read(out/'epochs.json'); selection=read(out/'selection.json')
    require(1<=len(epochs)<=int(row['max_epochs']),'Invalid epoch count')
    require([e['epoch'] for e in epochs]==list(range(1,len(epochs)+1)),'Epoch sequence gap')
    best=min(range(len(epochs)),key=lambda i:epochs[i]['validation_mse_std'])+1
    require(best==selection['best_epoch']==summary['best_epoch'] and not selection['test_labels_loaded'],
            'Checkpoint not first validation minimum')
    require(all(e['optimizer_steps']>0 and e['nonzero_gradient_parameters'] for e in epochs),'Missing training update')
    gradients=set(n for e in epochs for n in e['nonzero_gradient_parameters'])
    if row['pathway']!='history':
        require(any(n.startswith('backbone.') for n in gradients),'No observed backbone gradient')
        prefix='direct_head.' if row['pathway']=='direct' else 'adapter.'
        require(any(n.startswith(prefix) for n in gradients),'No observed fusion-head gradient')
    return predictions,metrics,scale,mean,cache


def pilot_gate(package=PACKAGE,results=RESULTS):
    verify_package(package)
    lock_hash=sha(package/'execution_lock.json')
    rows=[r for r in manifest(package) if pilot(r)]
    require(len(rows)==12,'Pilot count mismatch')
    for row in rows:
        check_predictions(row,package,results,lock_hash)
    gate=dict(status='PASS',scope='engineering_artifact_and_pairing_only',fits=12,
              accuracy_ranking_used=False,execution_lock_sha256=lock_hash,utc=utc(),
              run_summary_sha256={r['run_id']:sha(results/'runs'/r['run_id']/'summary.json') for r in rows})
    atomic_json(results/'pilot_gate.json',gate)
    print(json.dumps(gate),flush=True)


def aggregate_metrics(seed_rows):
    grouped=defaultdict(list)
    for row in seed_rows:
        grouped[(row['site'],row['horizon'],row['backbone'],row['pathway'])].append(row)
    stations=[]
    for (site,horizon,backbone,pathway),rows in sorted(grouped.items()):
        require({r['seed'] for r in rows}=={2024,2025,2026} and len(rows)==3,'Missing/duplicate seeds')
        row=dict(site=site,horizon=horizon,backbone=backbone,pathway=pathway,seeds=3)
        for k in METRICS:
            values=[r[k] for r in rows]
            row[k+'_mean']=None if any(v is None for v in values) else float(np.mean(values))
            row[k+'_sd']=None if any(v is None for v in values) else float(np.std(values,ddof=1))
        stations.append(row)
    tasks=[]
    groups=defaultdict(list)
    for row in stations:
        groups[(row['horizon'],row['backbone'],row['pathway'])].append(row)
    for (horizon,backbone,pathway),rows in sorted(groups.items()):
        require({r['site'] for r in rows}=={'2107','9068','14200'} and len(rows)==3,'Missing sites')
        row=dict(horizon=horizon,backbone=backbone,pathway=pathway,sites=3,aggregation='equal_site_seed_first')
        for k in METRICS:
            values=[r[k+'_mean'] for r in rows]
            for name,fn in (('mean',np.mean),('observed_min',np.min),('observed_max',np.max)):
                row[k+'_'+name]=None if any(v is None for v in values) else float(fn(values))
        tasks.append(row)
    return stations,tasks


def analyze_all(package=PACKAGE,results=RESULTS):
    verify_package(package)
    lock_hash=sha(package/'execution_lock.json')
    rows=manifest(package)
    out=results/'analysis'
    if (out/'summary.json').exists():
        done=read(out/'summary.json')
        require(done['status']=='PASS' and done['execution_lock_sha256']==lock_hash,'Existing analysis identity mismatch')
        for name,expected in done['artifacts_sha256'].items():
            require(sha(out/name)==expected,'Existing analysis changed: '+name)
        for row in rows:
            verify_run(results/'runs'/row['run_id'],row,lock_hash)
        print('VERIFIED_EXISTING_ANALYSIS_PASS',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    seed_rows=[]; raw_rows=[]; lead_rows=[]; baseline_rows=[]; losses={}; seen_controls=set()
    for row in rows:
        pred,metrics,scale,mean,cache=check_predictions(row,package,results,lock_hash)
        identity=dict(run_id=row['run_id'],site=row['site'],horizon=int(row['horizon_hours']),
                      backbone=row['backbone'],pathway=row['pathway'],seed=int(row['seed']))
        seed_rows.append(dict(identity,**metrics))
        raw_truth=(pred['raw_kw']-mean)/scale
        raw_rows.append(dict(identity,evaluation='same_clipped_trained_predictions_against_raw_AC',
                             **metric_values(pred['pred_std'],raw_truth,scale)))
        error=pred['pred_std']-pred['true_std']
        for lead in range(error.shape[1]):
            lead_rows.append(dict(identity,lead_hours=lead+1,
                                  mse_std=float(np.mean(error[:,lead]**2)),mae_std=float(np.mean(np.abs(error[:,lead])))))
        losses[(row['site'],identity['horizon'],row['backbone'],row['pathway'],identity['seed'])]=daily_loss(error,pred['origin_ns'])
        control_key=(row['site'],identity['horizon'])
        if control_key not in seen_controls:
            seen_controls.add(control_key)
            for name in ('persistence','climatology'):
                base=(cache[name+'_kw']-mean)/scale
                baseline_rows.append(dict(site=row['site'],horizon=identity['horizon'],control=name,
                                          **metric_values(base,pred['true_std'],scale)))
                atomic_npz(out/('%s_h%d_%s_predictions.npz'%(row['site'],identity['horizon'],name)),
                           pred_kw=cache[name+'_kw'],true_kw=pred['true_kw'],origin_ns=pred['origin_ns'],target_ns=pred['target_ns'])
    require(len(seed_rows)==108 and len(baseline_rows)==12,'Incomplete analysis matrix')
    stations,tasks=aggregate_metrics(seed_rows)
    by_station={(r['site'],r['horizon'],r['backbone'],r['pathway']):r for r in stations}
    paired=[]; paired_tasks=[]
    for site in ('2107','9068','14200'):
        for h in (4,24):
            for backbone in ('patchtst','itransformer'):
                for candidate,reference in PAIRS:
                    c=by_station[(site,h,backbone,candidate)]; r=by_station[(site,h,backbone,reference)]
                    for metric in ('mse_std','mae_std'):
                        cv,rv=c[metric+'_mean'],r[metric+'_mean']
                        paired.append(dict(site=site,horizon=h,backbone=backbone,candidate=candidate,reference=reference,
                                           metric=metric,candidate_seed_mean=cv,reference_seed_mean=rv,
                                           delta=cv-rv,relative_improvement=relative_improvement(cv,rv)))
    group=defaultdict(list)
    for row in paired:
        group[(row['horizon'],row['backbone'],row['candidate'],row['reference'],row['metric'])].append(row)
    for (h,b,c,r,m),group_rows in sorted(group.items()):
        relative=[x['relative_improvement'] for x in group_rows]
        delta=[x['delta'] for x in group_rows]
        paired_tasks.append(dict(horizon=h,backbone=b,candidate=c,reference=r,metric=m,sites=3,
                                 delta_mean=float(np.mean(delta)),observed_min=float(min(delta)),observed_max=float(max(delta)),
                                 mean_of_site_relative_improvements=None if any(v is None for v in relative) else float(np.mean(relative)),
                                 ratio_of_site_means_improvement=relative_improvement(
                                     float(np.mean([x['candidate_seed_mean'] for x in group_rows])),
                                     float(np.mean([x['reference_seed_mean'] for x in group_rows])))))
    intervals=[]
    for length in (7,14):
        weights=block_weights(length)
        atomic_npz(out/('bootstrap_calendar_weights_block%d.npz'%length),weights=weights)
        resampled={}
        for key,(squared,absolute,counts) in losses.items():
            resampled[key]=(resampled_loss(squared,counts,weights),resampled_loss(absolute,counts,weights))
        for site in ('2107','9068','14200'):
            for h in (4,24):
                for backbone in ('patchtst','itransformer'):
                    for candidate,reference in PAIRS:
                        for j,metric in enumerate(('mse_std','mae_std')):
                            delta=np.mean([resampled[(site,h,backbone,candidate,seed)][j]-
                                           resampled[(site,h,backbone,reference,seed)][j] for seed in (2024,2025,2026)],axis=0)
                            lo,hi=np.percentile(delta,[2.5,97.5])
                            point=by_station[(site,h,backbone,candidate)][metric+'_mean']-by_station[(site,h,backbone,reference)][metric+'_mean']
                            intervals.append(dict(site=site,horizon=h,backbone=backbone,candidate=candidate,reference=reference,
                                                  metric=metric,block_days=length,draws=10000,delta=point,lower=float(lo),upper=float(hi),
                                                  interpretation='conditional_within_site_descriptive_percentile'))
    for name,table in (('seed_metrics.csv',seed_rows),('site_seed_mean_sd.csv',stations),
                       ('task_equal_site_metrics.csv',tasks),('paired_site_effects.csv',paired),
                       ('paired_task_effects.csv',paired_tasks),('conditional_time_intervals.csv',intervals),
                       ('deterministic_controls.csv',baseline_rows),('raw_AC_evaluation_sensitivity.csv',raw_rows),
                       ('lead_metrics_per_seed.csv',lead_rows)):
        write_csv(out/name,table)
    failures=[dict(path=str(p.relative_to(results)),sha256=sha(p)) for p in sorted((results/'runs').glob('*/failure_*.json'))]
    atomic_json(out/'provenance.json',dict(execution_lock_sha256=lock_hash,
        run_summary_sha256={r['run_id']:sha(results/'runs'/r['run_id']/'summary.json') for r in rows},
        retained_failures=failures,human_scientific_verification='pending',
        target='max(hourly_measured_AC_kw,0)',nominal_init_not_operational_availability=True,
        no_general_population_inference=True,no_test_tuning=True))
    artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='summary.json'}
    summary=dict(status='PASS',scope='complete_locked_independent_validation_analysis',fits=108,
                 deterministic_control_cells=12,site_rows=len(stations),task_rows=len(tasks),interval_rows=len(intervals),
                 execution_lock_sha256=lock_hash,utc=utc(),artifacts_sha256=artifacts,
                 manuscript_integration='NOT_PERFORMED',human_scientific_review='PENDING')
    atomic_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-gate',action='store_true')
    parser.add_argument('--package', type=Path, required=True,
                        help='Authorized complete archived input package, including caches.')
    parser.add_argument('--results', type=Path, required=True,
                        help='Archived run artifacts; analysis output is created here if absent.')
    args=parser.parse_args()
    pilot_gate(args.package,args.results) if args.pilot_gate else analyze_all(args.package,args.results)
