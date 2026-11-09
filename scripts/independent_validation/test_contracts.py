"""Synthetic independent-data contracts and 12 CPU forwards, never training."""
import argparse
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import common
import analyze
import pvdaq_hrrr_window_loader_v1 as loader

CASES = []


class Contracts(unittest.TestCase):
    def example(self):
        origin = pd.Timestamp('2022-01-05')
        future = pd.date_range(origin+loader.HOUR,periods=4,freq='h')
        rows = pd.DataFrame(dict(origin_utc=[origin]*4,init_time_utc=[origin]*4,
                                lead_hours=[1,2,3,4],valid_time_utc=future,
                                pv_interval_start_utc=future-loader.HOUR,pv_interval_end_utc=future))
        return origin,rows

    def test_target_definition_preserves_raw(self):
        x = np.array([-3.,0.,12.])
        np.testing.assert_array_equal(loader.target_view(x,'zero_clipped_ac'),[0.,0.,12.])
        np.testing.assert_array_equal(x,[-3.,0.,12.])
        with self.assertRaises(ValueError):
            loader.target_view([np.nan],'zero_clipped_ac')

    def test_window_alignment_and_split_rejection(self):
        origin,rows = self.example()
        hist,future = loader.validate_window_times(origin,4,'test',rows)
        self.assertEqual((hist[0],hist[-1],future[0]),(origin-95*loader.HOUR,origin,origin+loader.HOUR))
        with self.assertRaisesRegex(ValueError,'split boundary'):
            loader.validate_window_times(pd.Timestamp('2022-01-02'),4,'test',rows)
        for field,value in [('init_time_utc',origin-loader.HOUR),('lead_hours',12),
                            ('valid_time_utc',origin),('pv_interval_start_utc',origin)]:
            bad = rows.copy(); bad.loc[1,field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                loader.validate_window_times(origin,4,'test',bad)

    def test_train_only_scaler(self):
        power = pd.Series([-4.,2.,6.,999.],index=pd.to_datetime([
            '2020-01-01 01:00','2020-01-01 02:00','2021-10-01 00:00','2021-10-01 01:00']))
        joined = pd.DataFrame(dict(split=['train','train','test'],
            init_time_utc=pd.to_datetime(['2020-01-01']*2+['2022-01-01']),lead_hours=[1,2,1],
            **{f:[1.,3.,999.] for f in loader.FIELDS}))
        before = loader.fit_site_scalers(power,joined,'zero_clipped_ac')
        power.iloc[-1] = -1e15
        joined.loc[joined.split.eq('test'),list(loader.FIELDS)] = -1e15
        joined = pd.concat([joined,joined.loc[joined.split.eq('train')]],ignore_index=True)
        self.assertEqual(before,loader.fit_site_scalers(power,joined,'zero_clipped_ac'))
        self.assertEqual(before['power']['rows'],3)
        self.assertEqual(before['nwp']['rows'],2)
        self.assertAlmostEqual(before['power']['mean'][0],8/3)
        bad = joined.iloc[[0]].copy(); bad.loc[:,'TMP'] = 5.
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            loader.fit_site_scalers(power,pd.concat([joined,bad]),'zero_clipped_ac')

    def test_common_history_and_no_future_label_input(self):
        origin,rows = self.example()
        hist,future = loader.validate_window_times(origin,4,'test',rows)
        outputs = [loader.pack_model_inputs(np.arange(96)[:,None],np.arange(28).reshape(4,7),
                                           hist,future,p) for p in loader.PATHWAYS]
        for out in outputs:
            np.testing.assert_array_equal(out['x_enc'],outputs[0]['x_enc'])
            self.assertFalse(out['x_enc'][:,:7].any())
            self.assertFalse(out['x_dec'][:,-1].any())
        self.assertFalse(outputs[0]['x_dec'].any())
        np.testing.assert_array_equal(outputs[1]['x_dec'],outputs[2]['x_dec'])

    def test_named_metrics_and_units(self):
        result = common.metric_values(np.array([0.,4.]),np.array([1.,1.]),2.)
        self.assertEqual(result['mse_std'],5.)
        self.assertEqual(result['mae_std'],2.)
        self.assertEqual(result['mae_kw'],4.)
        self.assertIsNone(result['r2'])

    def test_block_resampling(self):
        a = analyze.block_weights(7,draws=20)
        b = analyze.block_weights(7,draws=20)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((a.sum(axis=1)==365).all())
        with self.assertRaises(ValueError):
            analyze.resampled_loss(np.zeros(365),np.zeros(365),a)

    def test_missing_inputs_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            common.verify_package(common.PROJECT/'data/NO_SUCH_SYNTHETIC_PACKAGE')

    def test_no_training_without_opt_in(self):
        cmd = [sys.executable,'-B',str(common.HERE/'train_one.py'),'--run-id','synthetic',
               '--physical-gpu','0','--package','missing','--results','missing-output']
        result = subprocess.run(cmd,capture_output=True,encoding='utf-8',
                                env=dict(os.environ,PYTHONIOENCODING='utf-8'),timeout=60)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--allow-training',result.stderr)

    def test_models_and_future_target_invariance(self):
        import torch
        torch.set_num_threads(2); torch.manual_seed(2024)
        cfg = common.read(common.PROJECT/'configs/independent_validation.json')
        lock = common.read(common.PROJECT/'configs/independent_execution_lock.json')
        modules,_ = common.load_models(lock['expected_model_sources'])
        rows = [r for r in cfg['package_tables']['planned_run_manifest.csv']['records']
                if r['site']=='2107' and r['seed']=='2024']
        self.assertEqual(len(rows),12)
        for row in rows:
            h = int(row['horizon_hours'])
            model = modules[row['model']].Model(common.model_config(row)).cpu().eval()
            x = torch.zeros(2,96,8); x[:,:,-1] = torch.randn(2,96)
            dec = torch.zeros(2,h,8)
            if row['pathway'] != 'history':
                dec[:,:,:7] = torch.randn(2,h,7)
            changed = dec.clone(); changed[:,:,-1] = 10000.
            xm,ym = torch.randn(2,96,4),torch.randn(2,h,4)
            state = {k:v.clone() for k,v in model.state_dict().items()}
            with torch.no_grad():
                first,second = model(x,xm,dec,ym),model(x,xm,changed,ym)
            self.assertEqual(tuple(first.shape),(2,h,8))
            self.assertTrue(torch.isfinite(first).all().item())
            self.assertTrue(torch.equal(first,second))
            self.assertTrue(all(torch.equal(state[k],v) for k,v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            CASES.append(dict(backbone=row['backbone'],pathway=row['pathway'],horizon_hours=h,
                              finite=True,future_target_invariant=True,state_unchanged=True,
                              parameters=sum(p.numel() for p in model.parameters())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite a report.')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    import torch
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                  model_cases=CASES,python=platform.python_version(),torch=torch.__version__,
                  numpy=np.__version__,pandas=pd.__version__,training_performed=False,
                  scope='Synthetic CPU/data-contract checks, not CUDA accuracy replay')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
