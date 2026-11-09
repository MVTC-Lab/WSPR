"""Print one explicit paper-setting training command. Never execute it."""
import argparse
import json
import shlex
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['PVOD', 'GEFCom'], required=True)
    p.add_argument('--backbone', choices=['patchtst', 'itransformer'], required=True)
    p.add_argument('--pathway', choices=['history', 'direct', 'residual'], required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--seed', type=int, choices=[2024, 2025, 2026], default=2024)
    a = p.parse_args()
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/primary_matrix.json').read_text())
    ds = cfg['datasets'][a.dataset]
    if str(a.horizon) not in ds['tasks']:
        p.error('Horizon is not in this dataset paper matrix.')
    model = cfg['models'][a.backbone][a.pathway]
    params = dict(cfg['shared'])
    params.update(task_name='long_term_forecast', is_training=1, data='custom', features='MS',
                  target='OT', freq=ds['freq'], root_path=ds['root'], data_path=ds['file'],
                  model=model, model_id='release_example', seq_len=ds['tasks'][str(a.horizon)],
                  pred_len=a.horizon, fix_seed=a.seed, enc_in=ds['known_channels']+1,
                  dec_in=ds['known_channels']+1, c_out=ds['known_channels']+1,
                  use_future_covariates=int(a.pathway != 'history'),
                  known_future_dim=0 if a.pathway == 'history' else ds['known_channels'],
                  checkpoints='outputs/checkpoints', des='release_example', itr=1,
                  wspr_ablation='none', physics_loss_weight=0.0, reliability_loss_weight=0.0)
    cmd = ['python', 'run.py']
    for key, value in params.items():
        cmd.extend(['--' + key, str(value)])
    cmd.extend(['--use_amp', '--allow_training'])
    print('PROPOSAL ONLY: no process is launched. Check exact archived configuration before claiming reproduction.')
    print('This example uses the first site/region and will TRAIN if you execute it.')
    print(' '.join(shlex.quote(x) for x in cmd))

if __name__ == '__main__':
    main()
