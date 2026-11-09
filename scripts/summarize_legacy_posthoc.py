"""Reaggregate three post-hoc scalar analyses; no model, data download or fits."""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def key(row, columns):
    return tuple(row[c] for c in columns)


def verify(study, columns, site_table, task_table, expected_rows, expected_tasks, metrics):
    tables = study['tables']
    rows = tables['run_metrics.csv']['records']
    require(len(rows) == expected_rows, 'Wrong scalar row count')
    groups = defaultdict(list)
    for row in rows:
        groups[key(row, columns)+ (row['unit'],)].append(row)
    sites = tables[site_table]['records']
    indexed = {key(r,columns)+(r['unit'],):r for r in sites}
    require(len(indexed) == len(sites) and set(indexed) == set(groups), 'Site group mismatch')
    tasks = defaultdict(list)
    for identity, values in groups.items():
        require(len(values) == 3 and {r['seed'] for r in values} == {'2024','2025','2026'},
                'Incomplete/duplicate seeds')
        averages = {}
        for metric in metrics:
            numbers = [float(v[metric]) for v in values]
            require(all(math.isfinite(v) for v in numbers), 'Nonfinite scalar')
            averages[metric] = mean(numbers)
            require(math.isclose(averages[metric],float(indexed[identity][metric]),
                                 rel_tol=1e-10,abs_tol=1e-12), 'Seed average mismatch')
        tasks[identity[:-1]].append(averages)
    recorded = tables[task_table]['records']
    task_index = {key(r,columns):r for r in recorded}
    require(len(recorded) == len(task_index) == expected_tasks and set(task_index) == set(tasks),
            'Task group mismatch')
    for identity, values in tasks.items():
        require(len(values) == 10, 'Not all ten PVOD sites included')
        for metric in metrics:
            actual = mean(v[metric] for v in values)
            require(math.isclose(actual,float(task_index[identity][metric]),rel_tol=1e-10,abs_tol=1e-12),
                    'Equal-site task mean mismatch')
    return dict(status='PASS',scalar_rows=len(rows),site_rows=len(sites),task_rows=len(recorded))


def summarize():
    studies = json.loads((ROOT/'results/legacy_posthoc_tables.json').read_text(encoding='utf-8'))['studies']
    results = {}
    results['calendar_climatology'] = verify(studies['calendar_climatology'],('method','pred_len'),
        'station_seed_metrics.csv','task_metrics.csv',300,10,('mse','mae','rmse','r2'))
    results['calendar_climatology']['new_fits'] = 120
    results['calendar_climatology']['reused_reference_fits'] = 180
    results['suffix'] = verify(studies['suffix'],('backbone','pathway','seq_len','pred_len','subset'),
        'cell_seed_averaged.csv','task_summary.csv',1800,60,('mse','mae','rmse','r2'))
    results['suffix']['frozen_checkpoints'] = 360
    results['active_generation'] = verify(studies['active_generation'],
        ('backbone','pathway','pred_len','threshold_pct_q99'),
        'site_seed_averaged_metrics.csv','task_metrics.csv',1440,48,
        ('mse','mae','rmse','nrmse_q99','nmae_q99'))
    results['active_generation']['frozen_checkpoints'] = 360
    return dict(status='PASS',studies=results,training_performed=False,
                scope='Seed-first equal-site scalar arithmetic; not prediction recomputation or interval replay')


if __name__ == '__main__':
    print(json.dumps(summarize(),indent=2))
