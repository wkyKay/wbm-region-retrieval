"""Frozen six-way ablation plan, real execution identities and source protection."""
from datetime import datetime
from pathlib import Path
import re

from f_region_structure.state import (TZ, now, record, signed, read_signed, seal, verified)
from wbm_retrieval.common import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'paper_c_ablation_v1'
BASELINE = 'primary'
KS = [1, 5, 10, 20, 50, 100, 500]
STAGES = ('valid', 'test', 'cross_test')
SETTINGS = {'ks': KS, 'tie_seed': 2026, 'checkpoint': 'best_nce', 'source_seed': 2026,
            'training_required': False, 'selection': 'all_prespecified_variants',
            'valid_role': 'implementation_and_protocol_gate_only',
            'test_parameter_search': False}
REPLAY_KEYS = {
    'valid': {
        'primary': 'f_region_area_fusion_f_region_area_fusion_valid_20261120T155825_0800_valid_add_a0p4_b0p25',
        'no_a': 'f_region_area_fusion_f_region_area_fusion_valid_20261120T155825_0800_valid_no_area_a0p4'},
    'test': {'primary': 'f_region_area_fusion_f_region_area_fusion_test_20261120T160426_0800_test_add_a0p4_b0p25'},
    'cross_test': {'primary': 'f_region_j_cross_f_region_j_cross_test_20261120T202638_0800_test_add_a0p4_b0p25'},
}


def grid():
    from .scoring import grid as scientific_grid
    return scientific_grid()


def evaluation_signature(context_signature, stage, name, spec=None):
    if stage not in STAGES or name not in grid() or not context_signature:
        raise ValueError('Unknown stage, variant or context')
    expected = grid()[name]
    if spec is not None and spec != expected:
        raise ValueError('Ablation specification differs from the frozen plan')
    return digest({'context': context_signature, 'stage': stage, 'variant': name, 'spec': expected})


def identity(output, stage, name, signature):
    if stage not in STAGES or name not in grid():
        raise ValueError('Unknown evaluation identity')
    output = Path(output)
    key = 'paper_c_ablation_' + re.sub(r'[^A-Za-z0-9_.-]', '_', output.name) + '_' + stage + '_' + name
    path = output/'identities'/(stage+'__'+name+'.json')
    if path.exists():
        old = read_json(path)
        stamp = old.get('started_at')
        if not isinstance(stamp, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00', stamp):
            raise ValueError('Evaluation start time must be a real Asia/Shanghai timestamp')
        started = datetime.fromisoformat(stamp)
    else:
        started = now()
    value = {'schema_version': 1, 'key': key,
             'run_id': started.strftime('%Y%m%dT%H%M%S%z')+'__'+key,
             'started_at': started.isoformat(), 'stage': stage,
             'split': 'valid' if stage == 'valid' else 'test',
             'protocol': 'coverage_cross_combination_v1' if stage == 'cross_test' else 'retrieval_v1',
             'variant': name, 'evaluation_signature': signature}
    record(path, value)
    return value


evaluation_identity = identity


def plan(context_signature, frozen_at):
    """The entire plan exists before the first valid score; no metric selects variants."""
    stamp = datetime.fromisoformat(frozen_at)
    if stamp.utcoffset() != now().utcoffset():
        raise ValueError('Plan requires an Asia/Shanghai execution timestamp')
    return signed({'schema_version': 1, 'version': VERSION,
        'context_signature': context_signature, 'frozen_at': frozen_at,
        'stages': list(STAGES), 'variants': list(grid()), 'specs': grid(),
        'settings': SETTINGS, 'replay_controls': REPLAY_KEYS,
        'evaluation_count': 18, 'new_registration_count': 14,
        'selection_rule': 'none; all six prespecified variants evaluated in every stage',
        'valid_role': 'verification gate, never performance-based screening',
        'test_parameter_search': False})


def protection(output, after=False):
    """Protect historical trees and pre-existing results, excluding only this run."""
    output = Path(output).resolve()
    path = output/'source_protection_before.json'
    if not after:
        if path.exists():
            return
        found = {}
        for base in ('artifacts', 'artificial', 'protocol', 'results'):
            for item in sorted((ROOT/base).rglob('*')):
                if not item.is_file() or item.is_symlink() or item.resolve().is_relative_to(output):
                    continue
                stat = item.stat()
                row = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                if stat.st_size < 1048576:
                    row['sha256'] = file_hash(item)
                found[str(item.relative_to(ROOT))] = row
        record(path, found)
        return
    original = read_json(path)
    for name, expected in original.items():
        item = ROOT/name
        if not item.is_file() or item.is_symlink():
            raise ValueError('Protected source disappeared: '+name)
        stat = item.stat()
        actual = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if 'sha256' in expected:
            actual['sha256'] = file_hash(item)
        if actual != expected:
            raise ValueError('Protected historical source changed: '+name)
    record(output/'source_protection_after.json', {'status': 'unchanged', 'files': len(original),
        'scope': 'pre-existing artifacts/artificial/protocol/results sizes and mtimes; files below 1MiB also SHA256; used sources hash-bound separately'})
