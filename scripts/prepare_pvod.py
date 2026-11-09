"""Prepare the paper's PVOD prefix, optionally verifying against frozen CSVs.

Requires provider-authorized raw data. No download, training, or source overwrite.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from preprocess_research_datasets import build_pvod, find_zip


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True,
                   help='New directory; existing output is never overwritten.')
    p.add_argument('--verify-against', type=Path,
                   help='Optional frozen PVOD_pre80 directory, read only.')
    a = p.parse_args()
    raw = a.raw_dir.resolve(strict=True)
    out = a.out_dir.resolve()
    if out == raw or out in raw.parents:
        p.error('Output cannot be the raw directory or its ancestor.')
    if out.exists():
        p.error('Output already exists; choose a new directory.')
    archive = find_zip(raw, ['metadata.csv', 'station00.csv', 'station09.csv'])
    out.mkdir(parents=True, exist_ok=False)
    summaries = build_pvod(raw, out)
    prefix = out / 'PVOD_pre80'
    prefix.mkdir()
    records = []
    expected = ['PVOD_station{:02d}_nwp.csv'.format(i) for i in range(10)]
    actual = sorted(p.name for p in (out / 'all_stations/PVOD').glob('*.csv'))
    if actual != expected:
        raise RuntimeError('Expected exactly station00 through station09.')
    for name in expected:
        source = out / 'all_stations/PVOD' / name
        frame = pd.read_csv(source)
        retained = frame.iloc[:int(len(frame) * .80)].copy()
        target = prefix / name
        retained.to_csv(target, index=False)
        record = {'file': name, 'prepared_rows': len(frame), 'retained_rows': len(retained),
                  'prepared_sha256': sha(source), 'prefix_sha256': sha(target),
                  'train_rows': int(len(retained) * .7),
                  'test_rows': int(len(retained) * .2)}
        record['validation_rows'] = len(retained) - record['train_rows'] - record['test_rows']
        if a.verify_against:
            frozen = a.verify_against / name
            other = pd.read_csv(frozen)
            same_structure = (list(retained.columns) == list(other.columns)
                              and retained.shape == other.shape)
            same_dates = same_structure and retained['date'].equals(other['date'])
            record['frozen_sha256'] = sha(frozen)
            record['byte_identical'] = sha(target) == sha(frozen)
            record['same_columns_shape_dates'] = bool(same_dates)
            record['max_abs_numeric_difference'] = (
                float(np.max(np.abs(pd.read_csv(target).iloc[:, 1:].to_numpy()
                                    - other.iloc[:, 1:].to_numpy()))) if same_dates else None)
        records.append(record)
    verified = (all(r['byte_identical'] for r in records) if a.verify_against else None)
    report = {'status': ('PASS' if verified else 'FAIL') if a.verify_against else 'PREPARED_NOT_VERIFIED',
              'raw_archive_sha256': sha(archive), 'numpy_version': np.__version__,
              'pandas_version': pd.__version__, 'frozen_byte_identity': verified,
              'training_performed': False, 'records': records, 'preparation_summaries': summaries}
    (out / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 1 if verified is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
