"""Immutable scientific identities and sealed, isolated artifacts."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import re

from wbm_retrieval.common import digest, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo('Asia/Shanghai')


def now():
    return datetime.now(TZ).replace(microsecond=0)


def record(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Immutable metadata changed: {path}')
    else:
        write_json(path, value)


def signed(value):
    return {**value, 'signature': digest(value)}


def read_signed(path):
    value = read_json(path)
    if value.get('signature') != digest({k:v for k,v in value.items() if k != 'signature'}):
        raise ValueError(f'Invalid signed metadata: {path}')
    return value


def seal(folder, signature):
    folder = Path(folder)
    files = {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p != folder / 'completed.json'}
    record(folder / 'completed.json', {'status':'completed', 'signature':signature, 'files':files})


def verified(folder, signature=None):
    folder = Path(folder)
    if not (folder / 'completed.json').exists():
        return False
    marker = read_json(folder / 'completed.json')
    if marker.get('status') != 'completed' or (signature is not None and marker['signature'] != signature):
        raise ValueError(f'Completion identity mismatch: {folder}')
    for name, expected in marker['files'].items():
        p = folder / name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.is_file() or file_hash(p) != expected:
            raise ValueError(f'Changed or missing completed artifact: {p}')
    return True


def identity(output, split, variant, signature):
    safe_output = re.sub(r'[^A-Za-z0-9_.-]', '_', output.name)
    key = 'f_region_structure_' + safe_output + '_' + split + '_' + variant
    path = output / 'identities' / f'{split}__{variant}.json'
    if path.exists():
        value = read_json(path)
        started = value.get('started_at','')
        if not isinstance(started,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00',started):
            raise ValueError('Invalid persisted evaluation start time')
        time = datetime.fromisoformat(started)
        if (value['key'] != key or value['evaluation_signature'] != signature
                or value.get('schema_version') != 1 or value.get('split') != split or value.get('variant') != variant
                or value.get('run_id') != time.strftime('%Y%m%dT%H%M%S%z')+'__'+key):
            raise ValueError('Resume evaluation identity changed')
        return value
    time = now()
    value = dict(schema_version=1, key=key, run_id=time.strftime('%Y%m%dT%H%M%S%z')+'__'+key,
                 started_at=time.isoformat(), split=split, variant=variant, evaluation_signature=signature)
    record(path, value)
    return value


def inventory(root, exclude):
    """All old artifact sizes/mtimes, plus SHA256 for small metadata and source files."""
    root, exclude = Path(root), Path(exclude).resolve()
    found = {}
    for base in ('artifacts', 'artificial', 'protocol'):
        for p in sorted((root / base).rglob('*')):
            if not p.is_file() or p.is_symlink() or p.resolve().is_relative_to(exclude):
                continue
            st = p.stat()
            row = {'size':st.st_size, 'mtime_ns':st.st_mtime_ns}
            if st.st_size < 1048576:
                row['sha256'] = file_hash(p)
            found[str(p.relative_to(root))] = row
    return found


def protection(output, after=False):
    path = Path(output) / 'source_protection_before.json'
    if not after:
        if not path.exists():
            record(path, inventory(ROOT, output))
        return
    original = read_json(path)
    changed = []
    for name, expected in original.items():
        p = ROOT / name
        if not p.is_file():
            changed.append(name)
            continue
        st = p.stat()
        actual = {'size':st.st_size, 'mtime_ns':st.st_mtime_ns}
        if 'sha256' in expected:
            actual['sha256'] = file_hash(p)
        if actual != expected:
            changed.append(name)
    if changed:
        raise ValueError(f'Protected historical artifacts changed: {changed[:10]}')
    record(Path(output) / 'source_protection_after.json', {'status':'unchanged', 'files':len(original),
        'scope':'sizes/mtime for all pre-existing artifacts/artificial/protocol; SHA256 additionally for files under 1MiB; used sources separately hash-bound'})
