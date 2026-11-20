"""Separate immutable identities for the new valid and test runs."""
from datetime import datetime
from pathlib import Path
import re

from f_region_structure.state import (now, protection, read_signed, record,
                                      seal, signed, verified)
from wbm_retrieval.common import read_json


def identity(output, split, variant, signature):
    if split not in ('valid', 'test'):
        raise ValueError('Only valid/test evaluation identities are supported')
    output = Path(output)
    key = 'f_region_area_fusion_' + re.sub(r'[^A-Za-z0-9_.-]', '_', output.name) + '_' + split + '_' + variant
    path = output / 'identities' / (split + '__' + variant + '.json')
    if path.exists():
        value = read_json(path)
        stamp = value.get('started_at', '')
        if not isinstance(stamp, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00', stamp):
            raise ValueError('Invalid persisted start time')
        time = datetime.fromisoformat(stamp)
        expected = dict(schema_version=1, key=key, split=split, variant=variant,
                        evaluation_signature=signature, started_at=stamp,
                        run_id=time.strftime('%Y%m%dT%H%M%S%z') + '__' + key)
        if value != expected:
            raise ValueError('Persisted evaluation identity changed')
        return value
    time = now()
    value = dict(schema_version=1, key=key, split=split, variant=variant,
                 evaluation_signature=signature, started_at=time.isoformat(),
                 run_id=time.strftime('%Y%m%dT%H%M%S%z') + '__' + key)
    record(path, value)
    return value
