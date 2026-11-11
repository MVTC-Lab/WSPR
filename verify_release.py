"""Offline package/hash/static-import and credential-pattern checks; no ML imports."""
import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = {'models', 'layers', 'utils', 'exp', 'data_provider'}
MODEL_HASHES = {
    'WSPR.py': '0c73f5c56e2cc51c5ea7509dca0844b858440a9b28ac5a538a3e8d2ed23e771f',
    'DirectNWP.py': 'e54157703e3a8f43d7748f5f280fca96399527d65af9eaac34887217f2ac92bf',
    'WSPR_iTransformer.py': '88671ca8c3daf6b203a3f4a579382ad10d1278d440f8dc9116b8ca946aec4e46',
    'DirectNWP_iTransformer.py': 'f0831af57ffaeee475d55966f5cf38d578e38e6b6545ad4f3fb048f174933b2c',
    'PatchTST.py': '29835d4fedddd3cbee26cc60d6a54c04513f5a9ca32e4a0d08de2c9e059084e5',
    'iTransformer.py': '7fdc721d041b0f8f63be8fa794ecd68422fd958c7c8d449026320fd9f368788e',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    issues = []
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    records = {r['path']: r for r in manifest['files']}
    expected = set(records) | {'RELEASE_MANIFEST.json'}
    # Verify a clone's Git candidates or a standalone review ZIP. Never inspect
    # an ancestor repository. Runtime/data artifacts are outside the ZIP manifest.
    if (ROOT / '.git').exists():
        top = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel'], text=True).strip()
        if Path(top).resolve() != ROOT:
            raise SystemExit('Refusing to inspect an ancestor repository.')
        listed = set(subprocess.check_output(
            ['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines())
        candidates = {name for name in listed if (ROOT / name).is_file()}
        mode = 'git-candidates'
    else:
        candidates = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                      if p.is_file() and not set(p.relative_to(ROOT).parts).intersection(
                          {'outputs', 'data', 'checkpoints', '__pycache__', '.venv'})}
        mode = 'standalone-archive'
    for name in sorted(candidates - expected):
        issues.append('Unreviewed Git candidate: ' + name)
    for name in sorted(expected - candidates):
        issues.append('Manifest entry is absent/ignored: ' + name)
    patterns = {
        'private-key': r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
        'github-token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'aws-access-key': r'\bAKIA[0-9A-Z]{16}\b',
        'credential-url': r'https?://[^\s/@:]+:[^\s/@]+@',
        'private-home-path': r'/home/[a-zA-Z0-9_-]+/',
        'windows-user-path': r'[A-Za-z]:[\\/]Users[\\/]',
        'ipv4-address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    python_files = 0
    archives_checked = 0
    for name in sorted(candidates & expected):
        path = ROOT / name
        if path.is_symlink() or ROOT not in path.resolve().parents:
            issues.append('Unsafe path: ' + name)
            continue
        raw = path.read_bytes()
        if len(raw) > 5 * 1024 * 1024:
            issues.append('Oversize candidate: ' + name)
        if name in records and digest(path) != records[name]['sha256']:
            issues.append('Hash mismatch: ' + name)
        if path.suffix.lower() == '.zip':
            archives_checked += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        issues.append('Archive CRC mismatch: ' + name)
                    members = archive.namelist()
                    if len(members) != len(set(members)):
                        issues.append('Duplicate archive member: ' + name)
                    for member in members:
                        pure = PurePosixPath(member)
                        if pure.is_absolute() or '..' in pure.parts or '\\' in member:
                            issues.append('Unsafe archive member: ' + name)
            except (zipfile.BadZipFile, RuntimeError, OSError):
                issues.append('Unreadable archive: ' + name)
            continue
        content = raw.decode('utf-8-sig')
        for label, pattern in patterns.items():
            if re.search(pattern, content):
                issues.append(label + ' pattern (manual review): ' + name)
        if path.suffix != '.py':
            continue
        python_files += 1
        tree = ast.parse(content, filename=name, feature_version=(3, 8))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split('.')
                if parts[0] in LOCAL_PACKAGES:
                    if len(parts) == 1:
                        targets = [ROOT / parts[0] / (a.name + '.py') for a in node.names]
                    else:
                        targets = [ROOT.joinpath(*parts).with_suffix('.py')]
                    for target in targets:
                        if not target.is_file():
                            issues.append('Missing local import in ' + name + ': ' + target.name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split('.')
                    if parts[0] in LOCAL_PACKAGES and len(parts) > 1:
                        if not ROOT.joinpath(*parts).with_suffix('.py').is_file():
                            issues.append('Missing local import in ' + name + ': ' + alias.name)
    for name, value in MODEL_HASHES.items():
        if digest(ROOT / 'models' / name) != value:
            issues.append('Frozen model changed: ' + name)
    cfg = json.loads((ROOT / 'configs/primary_matrix.json').read_text())
    count = sum(d['units'] * len(d['tasks']) * 2 * 3 * len(cfg['seeds'])
                for d in cfg['datasets'].values())
    if count != 468:
        issues.append('Primary matrix count mismatch')
    from summarize_primary import summarize
    registry_check = summarize()
    from summarize_independent import summarize as summarize_independent
    from summarize_legacy_posthoc import summarize as summarize_posthoc
    independent = summarize_independent()
    posthoc = summarize_posthoc()
    result = {'status': 'PASS' if not issues else 'FAIL', 'candidate_files': len(candidates),
              'inventory_mode': mode, 'registry_task_groups': registry_check['task_pathway_groups'],
              'python_files_parsed': python_files, 'archives_checked': archives_checked, 'frozen_model_hashes_checked': len(MODEL_HASHES),
              'primary_matrix_runs': count, 'total_bytes': sum((ROOT/n).stat().st_size for n in candidates),
              'independent_validation_fits': independent['fits'],
              'posthoc_scalar_checks': posthoc['status'],
              'training_performed': False, 'network_access': False,
              'runtime_import_test': 'NOT PERFORMED by this static checker',
              'secret_scan_scope': 'bounded patterns, not a guarantee of absence', 'issues': issues}
    print(json.dumps(result, indent=2))
    return 1 if issues else 0

if __name__ == '__main__':
    sys.exit(main())
