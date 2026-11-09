"""Run the unchanged Task-15 builders in a new, isolated output directory."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--verify-against', type=Path)
    a = p.parse_args()
    raw = a.raw_zip.resolve(strict=True)
    out = a.out_dir.resolve()
    if out.exists():
        p.error('Output already exists; choose a new directory.')
    out.mkdir(parents=True, exist_ok=False)
    common = ROOT / 'scripts/wspr_q1'
    subprocess.run([sys.executable, str(common / 'prepare_gefcom2014_solar.py'),
                    '--raw_zip', str(raw), '--task', '15', '--out_dir', str(out / 'legacy'),
                    '--report_dir', str(out / 'legacy_audit')], check=True)
    subprocess.run([sys.executable, str(common / 'prepare_gefcom2014_solar_deaccum_v1.py'),
                    '--raw_zip', str(raw), '--old_dir', str(out / 'legacy'),
                    '--out_dir', str(out / 'GEFCom2014Solar_deaccum_v1'),
                    '--audit_dir', str(out / 'corrected_audit')], check=True)
    records = []
    for zone in range(1, 4):
        name = 'GEFCom2014Solar_zone{:02d}_nwp.csv'.format(zone)
        target = out / 'GEFCom2014Solar_deaccum_v1' / name
        with target.open(encoding='utf-8-sig') as f:
            count = sum(1 for _ in csv.DictReader(f))
        record = {'file': name, 'rows': count, 'sha256': sha(target)}
        if a.verify_against:
            record['frozen_sha256'] = sha(a.verify_against / name)
            record['byte_identical'] = record['sha256'] == record['frozen_sha256']
        records.append(record)
    with (out / 'corrected_audit/accumulated_field_statistics.csv').open(encoding='utf-8-sig') as f:
        stats = list(csv.DictReader(f))
    negatives = {field: sum(int(s['negative_increments_before_nonnegative_clip'])
                           for s in stats if s['field'] == field)
                 for field in ['VAR169', 'VAR175', 'VAR178', 'VAR228']}
    verified = all(r['byte_identical'] for r in records) if a.verify_against else None
    report = {'status': ('PASS' if verified else 'FAIL') if a.verify_against else 'PREPARED_NOT_VERIFIED',
              'raw_archive_sha256': sha(raw), 'frozen_byte_identity': verified,
              'records': records, 'negative_increments_before_clipping': negatives,
              'training_performed': False}
    (out / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 1 if verified is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
