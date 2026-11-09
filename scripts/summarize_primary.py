"""Reaggregate the 468 frozen scalar records; no model/data loading or training."""
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MODEL_IDS = {
    'WSPR': 'TimeMixer_7',
    'DirectNWP': 'TimeMixer_7_DirectNWP_OptMatched',
    'WSPR_iTransformer': 'iTransformer_ResidualNWP',
    'DirectNWP_iTransformer': 'iTransformer_DirectNWP_OptMatched',
}


def summarize():
    registry = json.loads((ROOT / 'configs/primary_run_registry.json').read_text())
    cfg = json.loads((ROOT / 'configs/primary_matrix.json').read_text())
    records = registry['records']
    expected = set()
    for dataset, d in cfg['datasets'].items():
        units = ['station{:02d}'.format(i) for i in range(10)] if dataset == 'PVOD' else [
            'zone{:02d}'.format(i) for i in range(1, 4)]
        expected.update((dataset, b, int(h), p, u, s)
                        for b, h, p, u, s in product(cfg['models'], d['tasks'],
                        ['history', 'direct', 'residual'], units, cfg['seeds']))
    keys = [(r['dataset'], r['backbone'], r['pred_len'], r['pathway'], r['unit'], r['seed']) for r in records]
    assert len(records) == len(set(keys)) == 468 and set(keys) == expected
    per_unit = defaultdict(list)
    for r in records:
        assert all(math.isfinite(r[k]) for k in ['mse', 'mae', 'r2'])
        assert r['mse'] >= 0 and r['mae'] >= 0
        current_model = cfg['models'][r['backbone']][r['pathway']]
        assert r['model'] == HISTORICAL_MODEL_IDS.get(current_model, current_model)
        assert r['seq_len'] == cfg['datasets'][r['dataset']]['tasks'][str(r['pred_len'])]
        key = (r['dataset'], r['backbone'], r['pred_len'], r['pathway'], r['unit'])
        per_unit[key].append(r)
    tasks = defaultdict(list)
    for key, group in per_unit.items():
        assert {r['seed'] for r in group} == set(cfg['seeds'])
        tasks[key[:4]].append({metric: mean(r[metric] for r in group) for metric in ['mse', 'mae', 'r2']})
    output = []
    for key, group in sorted(tasks.items()):
        r = dict(zip(['dataset', 'backbone', 'pred_len', 'pathway'], key))
        r.update(n_units=len(group), **{k: mean(g[k] for g in group) for k in ['mse', 'mae', 'r2']})
        assert len(group) == cfg['datasets'][key[0]]['units']
        output.append(r)
    return {'status': 'PASS', 'primary_records': len(records), 'physical_unit_groups': len(per_unit),
            'task_pathway_groups': len(output), 'training_performed': False, 'tasks': output}


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
