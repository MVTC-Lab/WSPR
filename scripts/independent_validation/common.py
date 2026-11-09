"""Portable, torch-free utilities for the locked independent validation runner."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
# Release-only path relocation. No scientific settings or metric functions change.
PROJECT = HERE.parents[1]
PACKAGE = PROJECT / 'data/independent_validation/package'
RESULTS = PROJECT / 'outputs/independent_validation'
HISTORICAL_LOCK_SHA256 = 'cba9e6b5bde344d3ae56dd5826e42a692f69aa510a194a6a71ded75fe16152c9'
MODEL_ORDER = ['PatchTST', 'iTransformer', 'WSPR', 'DirectNWP',
               'DirectNWP_iTransformer', 'WSPR_iTransformer']
# Archived manifests keep their original model identifiers. These aliases let
# the renamed publication modules replay those records without rewriting history.
MODEL_ID_ALIASES = {
    'TimeMixer_7': 'WSPR',
    'TimeMixer_7_v8_3_final': 'WSPR',
    'TimeMixer_7_DirectNWP_OptMatched': 'DirectNWP',
    'iTransformer_DirectNWP_OptMatched': 'DirectNWP_iTransformer',
    'iTransformer_ResidualNWP': 'WSPR_iTransformer',
}
MODEL_SOURCE_KEYS = {
    'PatchTST': 'models/PatchTST.py',
    'iTransformer': 'models/iTransformer.py',
    'WSPR': 'models/TimeMixer_7.py',
    'DirectNWP': 'models/TimeMixer_7_DirectNWP_OptMatched.py',
    'DirectNWP_iTransformer': 'models/iTransformer_DirectNWP_OptMatched.py',
    'WSPR_iTransformer': 'models/iTransformer_ResidualNWP.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


def event(path, **record):
    record = dict(utc=utc(), **record)
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
    print(json.dumps(record, ensure_ascii=False), flush=True)


def write_csv(path, rows):
    require(bool(rows), 'Cannot write empty table: ' + str(path))
    fields = list(rows[0])
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest(package=PACKAGE):
    with (package / 'planned_run_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 108 and len({r['run_id'] for r in rows}) == 108, 'Invalid 108-run manifest')
    expected = {(s,h,b,p,k) for s in ('2107','9068','14200') for h in ('4','24')
                for b in ('patchtst','itransformer') for p in ('history','direct','residual')
                for k in ('2024','2025','2026')}
    require({(r['site'],r['horizon_hours'],r['backbone'],r['pathway'],r['seed']) for r in rows} == expected,
            'Manifest factorial mismatch')
    return rows


def pilot(row):
    return row['site'] == '2107' and row['seed'] == '2024'


def model_config(row):
    ints = ['enc_in','dec_in','c_out','d_model','d_ff','n_heads','e_layers','factor',
            'known_future_dim','use_future_covariates','label_len']
    floats = ['dropout','residual_l1_weight','physics_loss_weight']
    config = {k:int(row[k]) for k in ints}
    config.update({k:float(row[k]) for k in floats})
    config.update({k:row[k] for k in ['freq','embed','activation','wspr_ablation']})
    config.update(task_name='long_term_forecast', seq_len=int(row['history_hours']),
                  pred_len=int(row['horizon_hours']), reliability_target_idx=-1)
    return SimpleNamespace(**config)


def verify_package(package=PACKAGE, full=True):
    """Validate archived inputs plus separately hashed release adaptations.

    This does not assert byte identity of the relocated runner to the original
    execution scripts. The original execution lock remains unchanged. Private
    deployment scripts are excluded; all original input-artifact hashes remain
    mandatory, including cache hashes when full=True.
    """
    package = Path(package)
    require(sha(package / 'execution_lock.json') == HISTORICAL_LOCK_SHA256,
            'Not the archived independent-validation input package')
    lock = read(package / 'execution_lock.json')
    require(lock['training_authorized'] is True and lock['status'] == 'FROZEN_FOR_EXECUTION', 'No execution lock')
    release = read(PROJECT / 'RELEASE_MANIFEST.json')
    entries = {r['path']: r for r in release['files']}
    for name in ('common.py', 'train_one.py', 'analyze.py', 'pvdaq_hrrr_window_loader_v1.py'):
        rel = 'scripts/independent_validation/' + name
        require(rel in entries and sha(PROJECT/rel) == entries[rel]['sha256'],
                'Release implementation changed: ' + name)
    name = 'pvdaq_hrrr_window_loader_v1.py'
    require(sha(HERE/name) == lock['preparation_code_sha256'][name], 'Frozen loader changed')
    for name, expected in lock['package_sha256'].items():
        if full or not name.startswith('cache/'):
            require(sha(package/name) == expected, 'Package changed: ' + name)
    return lock


def load_cache(package, row, split):
    name = 'cache/%s_h%s_%s.npz' % (row['site'],row['horizon_hours'],split)
    lock = read(package/'execution_lock.json')
    require(sha(package/name) == lock['package_sha256'][name], 'Cache checksum mismatch')
    with np.load(package/name, allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    n, h = len(data['origin_ns']), int(row['horizon_hours'])
    require(n > 0 and data['x_enc'].shape == (n,96,8) and data['x_dec'].shape == (n,h,8), 'Cache shape')
    require(not data['x_enc'][:,:,:7].any() and not data['x_dec'][:,:,-1].any(), 'Unsafe cache input')
    require(np.isfinite(data['target_scaled']).all(), 'Nonfinite cache target')
    return data


def load_models(expected):
    sys.path.insert(0, str(PROJECT))
    for rel in ('models/__init__.py','layers/__init__.py','utils/__init__.py',
                'layers/Embed.py','layers/Transformer_EncDec.py','layers/SelfAttention_Family.py','utils/masking.py'):
        require(sha(PROJECT/rel) == expected[rel], 'Frozen model dependency mismatch: ' + rel)
    release = read(PROJECT / 'RELEASE_MANIFEST.json')
    entries = {r['path']: r for r in release['files']}
    import models
    modules = {}
    for name in MODEL_ORDER:
        rel = 'models/%s.py' % name
        source_rel = MODEL_SOURCE_KEYS[name]
        require(source_rel in expected, 'Historical model source missing from execution lock: ' + source_rel)
        require(rel in entries and entries[rel].get('source_sha256') == expected[source_rel],
                'Renamed model provenance mismatch: ' + rel)
        paths = [p/rel for p in (PROJECT,)
                 if (p/rel).is_file() and sha(p/rel) == entries[rel]['sha256']]
        require(bool(paths), 'Frozen model unavailable: ' + rel)
        spec = importlib.util.spec_from_file_location('models.'+name, paths[0])
        mod = importlib.util.module_from_spec(spec)
        sys.modules['models.'+name] = mod
        setattr(models, name, mod)
        spec.loader.exec_module(mod)
        modules[name] = mod
    modules.update({legacy: modules[current] for legacy, current in MODEL_ID_ALIASES.items()})
    imported = {}
    for name, mod in list(sys.modules.items()):
        if name.split('.')[0] not in ('models','layers','utils') or not getattr(mod,'__file__',None):
            continue
        path = Path(mod.__file__).resolve()
        rel = name.replace('.','/') + ('/__init__.py' if path.name == '__init__.py' else '.py')
        if rel in entries:
            verified = sha(path) == entries[rel]['sha256']
        else:
            verified = rel in expected and sha(path) == expected[rel]
        require(verified, 'Unverified imported source: ' + name)
        imported[name] = dict(path=str(path),sha256=sha(path))
    return modules, imported


def metric_values(prediction, truth, power_scale):
    p, t = np.asarray(prediction,dtype=np.float64), np.asarray(truth,dtype=np.float64)
    require(p.shape == t.shape and p.size > 0 and np.isfinite(p).all() and np.isfinite(t).all(), 'Invalid scoring arrays')
    error = p-t
    mse, mae = float(np.mean(error**2)), float(np.mean(np.abs(error)))
    denom = float(np.sum((t-t.mean())**2))
    return dict(mse_std=mse,mae_std=mae,rmse_std=float(np.sqrt(mse)),
                r2=None if denom == 0 else float(1-np.sum(error**2)/denom),
                mae_kw=mae*power_scale,rmse_kw=float(np.sqrt(mse))*power_scale)


def relative_improvement(candidate, reference):
    return None if reference == 0 else (reference-candidate)/reference


def next_lr(base, completed_epoch):
    return base * 0.5**(completed_epoch-1)


def epoch_order(size, seed, epoch):
    return np.random.RandomState(seed + 100000*epoch).permutation(size)


def verify_run(directory, row, lock_hash):
    summary = read(directory/'summary.json')
    require(summary['status'] == 'PASS' and summary['run_id'] == row['run_id'] and
            summary['execution_lock_sha256'] == lock_hash, 'Wrong completed run identity')
    for name, expected in summary['artifacts_sha256'].items():
        require(sha(directory/name) == expected, 'Run artifact mismatch: ' + str(directory/name))
    return summary
