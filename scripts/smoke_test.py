"""Non-training CPU runtime checks on synthetic data; not result reproduction."""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.output and args.output.exists():
        p.error('Refusing to overwrite an existing report.')
    import numpy as np
    import pandas as pd
    import torch
    from data_provider.data_loader import Dataset_Custom
    torch.set_num_threads(2)
    torch.manual_seed(2024)
    cfg = json.loads((ROOT / 'configs/primary_matrix.json').read_text())
    cases = []
    for dataset, ds in cfg['datasets'].items():
        for horizon, history in ds['tasks'].items():
            horizon = int(horizon)
            channels = ds['known_channels'] + 1
            for backbone, models in cfg['models'].items():
                for pathway, model_name in models.items():
                    settings = dict(cfg['shared'])
                    settings.update(task_name='long_term_forecast', seq_len=history,
                                    pred_len=horizon, enc_in=channels, dec_in=channels,
                                    c_out=channels, embed='timeF', freq=ds['freq'],
                                    factor=1, activation='gelu', wspr_ablation='none',
                                    use_future_covariates=int(pathway != 'history'),
                                    known_future_dim=0 if pathway == 'history' else channels-1)
                    model = importlib.import_module('models.' + model_name).Model(
                        SimpleNamespace(**settings)).cpu().eval()
                    x = torch.randn(2, history, channels)
                    future = torch.randn(2, horizon, channels)
                    tdim = 5 if ds['freq'] == 't' else 4
                    xm = torch.randn(2, history, tdim)
                    ym = torch.randn(2, horizon, tdim)
                    changed = future.clone()
                    changed[:, :, -1] += 10000.
                    with torch.no_grad():
                        out = model(x, xm, future, ym)
                        other = model(x, xm, changed, ym)
                    assert tuple(out.shape) == (2, horizon, channels)
                    assert torch.isfinite(out).all()
                    diff = float((out - other).abs().max())
                    assert diff == 0., (model_name, 'future target entered prediction', diff)
                    assert all(v.grad is None for v in model.parameters())
                    head = getattr(model, 'adapter', getattr(model, 'direct_head', None))
                    cases.append({'dataset': dataset, 'history': history, 'horizon': horizon,
                                  'backbone': backbone, 'pathway': pathway, 'model': model_name,
                                  'output_shape': list(out.shape), 'finite': True,
                                  'future_target_invariance_max_abs': diff,
                                  'parameters': sum(v.numel() for v in model.parameters()),
                                  'additional_head_parameters': (sum(v.numel() for v in head.parameters())
                                                                 if head is not None else 0)})
    output_root = ROOT / 'outputs'
    output_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-smoke-', dir=str(output_root)) as tmp:
        n, length, horizon = 1000, 96, 16
        numeric = np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float)*2+1])
        df = pd.DataFrame(numeric, columns=['known', 'OT'])
        df.insert(0, 'date', pd.date_range('2020-01-01', periods=n, freq='15min'))
        df.to_csv(Path(tmp) / 'synthetic.csv', index=False)
        counts = {}
        for flag in ['train', 'val', 'test']:
            ds = Dataset_Custom(SimpleNamespace(augmentation_ratio=0, unit_interval_cols=''),
                                root_path=tmp, flag=flag, size=[length, 0, horizon],
                                features='MS', data_path='synthetic.csv', target='OT',
                                timeenc=1, freq='t')
            assert np.allclose(ds.scaler.mean_, numeric[:700].mean(axis=0))
            first_target = ds.scaler.inverse_transform(ds[0][1])[0, -1]
            first_index = {'train': length, 'val': 700, 'test': 800}[flag]
            assert np.isclose(first_target, numeric[first_index, -1])
            counts[flag] = len(ds)
        assert counts == {'train': 589, 'val': 85, 'test': 185}, counts
    child_env = dict(os.environ, PYTHONIOENCODING='utf-8')
    help_run = subprocess.run([sys.executable, '-B', 'run.py', '--help'], cwd=str(ROOT),
                              capture_output=True, encoding='utf-8', env=child_env, timeout=90)
    assert help_run.returncode == 0 and '--allow_training' in help_run.stdout
    guard = subprocess.run([sys.executable, '-B', 'run.py', '--data', 'custom',
                            '--is_training', '1'], cwd=str(ROOT),
                           capture_output=True, encoding='utf-8', env=child_env, timeout=90)
    assert guard.returncode != 0 and '--allow_training' in (guard.stdout + guard.stderr)
    report = {'status': 'PASS', 'scope': 'synthetic CPU forward, split/scaling, CLI and training guard',
              'python': platform.python_version(), 'torch': torch.__version__,
              'numpy': np.__version__, 'pandas': pd.__version__,
              'model_task_cases': len(cases), 'cases': cases, 'synthetic_window_counts': counts,
              'cli_help': 'PASS', 'training_guard': 'PASS', 'training_performed': False,
              'backward_or_optimizer_step': False,
              'limitations': 'Synthetic CPU checks only; environment installation and checkpoint replay require separate evidence.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
