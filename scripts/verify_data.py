"""Read-only SHA256 comparison of prepared inputs to the frozen run registry."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['PVOD', 'GEFCom'], required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    a = p.parse_args()
    records = json.loads((ROOT / 'configs/primary_run_registry.json').read_text())['records']
    expected = {}
    for r in records:
        if r['dataset'] != a.dataset:
            continue
        name = r['data_file']
        if name in expected:
            assert expected[name] == r['data_sha256']
        expected[name] = r['data_sha256']
    output = []
    for name, value in sorted(expected.items()):
        path = a.data_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        output.append({'file': name, 'expected_sha256': value, 'actual_sha256': actual,
                       'match': actual == value})
    passed = bool(output) and all(r['match'] for r in output)
    print(json.dumps({'status': 'PASS' if passed else 'FAIL', 'dataset': a.dataset,
                      'records': output, 'training_performed': False}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
