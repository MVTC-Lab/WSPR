"""Reaggregate all 108 independent fits from frozen scalar tables; no training.

This validates arithmetic and pairing, not predictions, accuracy or causal claims.
CSV numeric strings are explicitly converted by metric name (never by column index).
"""
import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('mse_std', 'mae_std', 'rmse_std', 'r2', 'mae_kw', 'rmse_kw')
LOCK = 'cba9e6b5bde344d3ae56dd5826e42a692f69aa510a194a6a71ded75fe16152c9'


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    if a is None or b in ('', None):
        require(a is None and b in ('', None), 'Missing-value mismatch')
    else:
        require(math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12),
                'Frozen scalar arithmetic mismatch')


def summarize():
    cfg = json.loads((ROOT/'configs/independent_validation.json').read_text(encoding='utf-8'))
    evidence = json.loads((ROOT/'results/independent_validation.json').read_text(encoding='utf-8'))
    require(cfg['historical_execution_lock_sha256'] == evidence['historical_execution_lock_sha256'] == LOCK,
            'Execution identity mismatch')
    require(hashlib.sha256((ROOT/'configs/independent_execution_lock.json').read_bytes()).hexdigest() == LOCK,
            'Historical lock has changed')
    rows = evidence['tables']['seed_metrics.csv']['records']
    planned = cfg['package_tables']['planned_run_manifest.csv']['records']
    require(len(rows) == len(planned) == 108, 'Incomplete 108-run matrix')
    require(len({r['run_id'] for r in rows}) == 108 and
            {r['run_id'] for r in rows} == {r['run_id'] for r in planned}, 'Run identity mismatch')
    expected = set(product(cfg['sites'], ('4','24'), ('patchtst','itransformer'),
                           ('history','direct','residual'), ('2024','2025','2026')))
    require({(r['site'],r['horizon'],r['backbone'],r['pathway'],r['seed']) for r in rows} == expected,
            'Factorial design mismatch')
    by_id = {r['run_id']: r for r in planned}
    groups = defaultdict(list)
    for row in rows:
        plan = by_id[row['run_id']]
        for key in ('site','backbone','pathway','seed'):
            require(row[key] == plan[key], 'Manifest identity mismatch')
        require(row['horizon'] == plan['horizon_hours'] and plan['history_hours'] == '96' and
                plan['target_mode'] == 'zero_clipped_ac' and plan['history_information'] == 'power_only',
                'Unexpected independent data contract')
        values = {k: None if row[k] == '' else float(row[k]) for k in METRICS}
        require(all(v is None or math.isfinite(v) for v in values.values()), 'Nonfinite metric')
        require(all(values[k] >= 0 for k in METRICS if k != 'r2'), 'Negative error metric')
        close(values['rmse_std']**2, values['mse_std'])
        require(values['mae_std'] <= values['rmse_std'] + 1e-12, 'MAE exceeds RMSE')
        scale = cfg['train_only_scalers'][row['site']]['power']['scale'][0]
        close(values['mae_std']*scale, values['mae_kw'])
        close(values['rmse_std']*scale, values['rmse_kw'])
        groups[(row['site'],row['horizon'],row['backbone'],row['pathway'])].append(values)
    recorded_sites = evidence['tables']['site_seed_mean_sd.csv']['records']
    require(len(recorded_sites) == 36 and len(groups) == 36, 'Missing site rows')
    site_index = {(r['site'],r['horizon'],r['backbone'],r['pathway']):r for r in recorded_sites}
    require(set(site_index) == set(groups), 'Site index mismatch')
    tasks = defaultdict(list)
    for key, values in groups.items():
        site = {}
        for metric in METRICS:
            numbers = [v[metric] for v in values]
            site[metric] = None if None in numbers else mean(numbers)
            close(site[metric],site_index[key][metric+'_mean'])
            close(None if None in numbers else stdev(numbers),site_index[key][metric+'_sd'])
        tasks[key[1:]].append(site)
    recorded_tasks = evidence['tables']['task_equal_site_metrics.csv']['records']
    require(len(recorded_tasks) == 12, 'Missing task rows')
    task_index = {(r['horizon'],r['backbone'],r['pathway']):r for r in recorded_tasks}
    require(set(task_index) == set(tasks), 'Task index mismatch')
    result = []
    for key, values in sorted(tasks.items()):
        task = dict(horizon_hours=int(key[0]),backbone=key[1],pathway=key[2],sites=3)
        for metric in METRICS:
            numbers = [v[metric] for v in values]
            for label, fn in [('mean',mean),('observed_min',min),('observed_max',max)]:
                estimate = None if None in numbers else fn(numbers)
                close(estimate,task_index[key][metric+'_'+label])
                if label == 'mean':
                    task[metric] = estimate
        result.append(task)
    leads = defaultdict(list)
    for row in evidence['tables']['lead_metrics_per_seed.csv']['records']:
        leads[row['run_id']].append(row)
    for row in rows:
        lead_rows = leads[row['run_id']]
        h = int(row['horizon'])
        require(len(lead_rows) == h and {int(x['lead_hours']) for x in lead_rows} == set(range(1,h+1)),
                'Missing/duplicate lead')
        for metric in ('mse_std','mae_std'):
            close(mean(float(x[metric]) for x in lead_rows),row[metric])
    paired = evidence['tables']['paired_task_effects.csv']['records']
    require(len(paired) == 24, 'Missing paired comparisons')
    for row in paired:
        key = (row['horizon'],row['backbone'])
        metric = row['metric']
        c = float(task_index[key+(row['candidate'],)][metric+'_mean'])
        r = float(task_index[key+(row['reference'],)][metric+'_mean'])
        close(c-r,row['delta_mean'])
        close((r-c)/r,row['ratio_of_site_means_improvement'])
    return {'status':'PASS','fits':108,'site_pathway_rows':36,'task_pathway_rows':12,
            'all_lead_rows':sum(len(x) for x in leads.values()),'paired_task_rows':24,
            'scope':'Frozen scalar arithmetic; not prediction recomputation or training',
            'tasks':result}


if __name__ == '__main__':
    print(json.dumps(summarize(),indent=2,allow_nan=False))
