"""Validate the sealed valid choice before loading its read-only test cache."""
from pathlib import Path

import numpy as np

from f_region_score_search import inputs as valid_inputs
from f_region_score_search import scoring, register as valid_register
from f_region_structure import sources
from f_region_structure.state import read_signed, signed, verified
from wbm_retrieval.common import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'artificial/f_region_score_search_20261120T140544+0800'
VARIANT = 'mixed_a0p5_e1p0'
SPEC = scoring.frozen_grid()[VARIANT]


def frozen_choice(source):
    """Validate the complete previous valid search, not a test-driven subset."""
    source = Path(source).resolve()
    manifest = read_signed(source/'run_manifest.json')
    signature = digest(manifest['context'])
    if not verified(source, signature):
        raise ValueError('The complete valid search must be sealed')
    proposed = valid_register.proposal(source)
    if len(proposed['runs']) != 15:
        raise ValueError('Expected all 15 valid candidates')
    selected = read_signed(source/'selection.json')
    if selected.get('winner') != VARIANT or selected.get('selected_spec') != SPEC:
        raise ValueError('Only the frozen equal-weight valid winner is authorized')
    for name, expected in manifest['context']['code'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError('Frozen valid scientific code changed: '+name)
    value = signed({'schema_version': 1, 'source_run': str(source),
        'source_selection_sha256': file_hash(source/'selection.json'),
        'source_completed_sha256': file_hash(source/'completed.json'),
        'source_valid_index_sha256': file_hash(source/'evaluation_index_valid.json'),
        'selection': selected, 'winner': VARIANT, 'selected_spec': SPEC,
        'selection_policy': 'frozen_valid_winner_only'})
    binding = {}
    valid_inputs._bind(source/'completed.json', binding)
    for name, expected in read_json(source/'completed.json')['files'].items():
        valid_inputs._bind(source/name, binding, expected)
    return manifest, value, binding


def _check_cache(cache, protocol):
    ids = valid_inputs._integer_array(cache['map_ids'], (7199,), 'map_ids', unique=True)
    pool = protocol.retrieval.test
    if (pool.split_name != 'test' or len(pool) != 7199 or len(pool.gallery) != 7199
            or pool.candidates_per_query != 7198
            or not np.array_equal(ids, protocol.ids['test'])
            or set(map(int,pool)) != set(map(int,ids))
            or set(map(int,pool.gallery)) != set(map(int,ids))):
        raise ValueError('Cache must match the frozen 7199-query full test protocol')
    offsets = valid_inputs._integer_array(cache['offsets'], (7200,), 'offsets')
    counts = np.diff(offsets.astype(np.int64))
    if offsets[0] != 0 or np.any(counts < 0) or np.any(counts > 4):
        raise ValueError('Invalid frozen Top-4 offsets')
    n = int(offsets[-1])
    arrays = {name: valid_inputs._integer_array(cache[name], (n,), name)
              for name in ('area','original_count','retained_count','owners')}
    areas, retained = arrays['area'], arrays['retained_count']
    if (not n or np.any(areas < 5) or not np.array_equal(areas, arrays['original_count'])
            or np.any(retained > areas) or not np.array_equal(arrays['owners'], np.repeat(ids, counts))):
        raise ValueError('Frozen region ownership or original areas differ')
    h = np.asarray(cache['features'])
    if h.shape != (n,512) or h.dtype.kind != 'f' or not np.isfinite(h).all():
        raise ValueError('Invalid frozen h512 features')
    norm = np.linalg.norm(h.astype(np.float64),axis=1)
    if np.any((norm != 0) & ~np.isclose(norm,1.,rtol=1e-5,atol=1e-6)):
        raise ValueError('CNN features must be normalized or zero')
    radial = valid_inputs._probabilities(cache['histograms'], (n,16), 'radial')
    if not np.array_equal(radial,cache['radial_hist16']):
        raise ValueError('Radial descriptors disagree')
    relative = np.asarray(cache['relative_area'])
    if relative.shape != (n,) or not np.isfinite(relative).all() or np.any((relative <= 0)|(relative > 1)):
        raise ValueError('Invalid original-mask relative area')
    for suffix, count, may_empty in (('',retained,True),('_all',areas,False)):
        joint = valid_inputs._probabilities(cache['joint'+suffix], (n,128), 'joint'+suffix, empty=may_empty)
        shape = valid_inputs._probabilities(cache['shape'+suffix], (n,16), 'shape'+suffix, empty=may_empty)
        flags = np.asarray(cache['empty'+suffix])
        if (flags.shape != (n,) or flags.dtype != bool or not np.array_equal(flags,count == 0)
                or not np.array_equal(joint.sum(1)==0,flags)
                or not np.allclose(shape,joint.reshape(n,8,4,4).sum(1).reshape(n,16),rtol=0,atol=1e-12)):
            raise ValueError('Structure marginal/count/empty flags disagree')
    quality = np.asarray(cache['quality'])
    if quality.shape != (n,) or not np.isfinite(quality).all() or not np.allclose(quality,retained/areas,rtol=0,atol=1e-12):
        raise ValueError('Region quality differs from point counts')


def load(source=DEFAULT_SOURCE):
    source = Path(source).resolve()
    manifest, choice, binding = frozen_choice(source)
    old_run = Path(manifest['source_run']).resolve()
    data = valid_inputs.load(old_run)
    if (data['source_binding'] != manifest['context'].get('source_binding')
            or data['config']['matching'] != manifest['context'].get('matching')):
        raise ValueError('Test inputs must use the identical model/protocol/matching bound by valid')
    for name, expected in data['source_binding'].items():
        valid_inputs._bind(Path(name), binding, expected)
    # Reuse the same original model, descriptor and cache identity selected on valid.
    old_manifest = read_signed(old_run/'run_manifest.json')
    folder = old_run/'cache/test'
    signature = digest({'context': data['old_context_signature'], 'split':'test','cache_version':1})
    valid_inputs._sealed(folder, signature, valid_inputs.CACHE_FILES, binding)
    if (folder/'h_views.npy').exists():
        raise ValueError('Test must not contain unbound training views')
    upstream = read_json(folder/'source_binding.json')
    roots = [(Path(p).resolve()/'test') for p in old_manifest['feature_roots']]
    if not isinstance(upstream,dict) or not upstream:
        raise ValueError('Test cache is missing upstream bindings')
    for name, expected in upstream.items():
        path = Path(name)
        if not path.is_absolute() or not any(path.resolve().is_relative_to(root) for root in roots):
            raise ValueError('Cache dependency must be in its declared test feature source')
        valid_inputs._bind(path,binding,expected)
    cache = sources.read_cache(folder)
    _check_cache(cache,data['protocol'])
    audit = read_json(folder/'audit.json')
    if (audit.get('split') != 'test' or audit.get('maps') != 7199
            or audit.get('regions') != int(cache['offsets'][-1])
            or audit.get('map_ids') != cache['map_ids'].tolist()):
        raise ValueError('Test descriptor audit and cache identities differ')
    data.update(cache=cache, source_binding=dict(sorted(binding.items())),
                frozen_selection=choice, source_run=source, cache_source_run=old_run)
    return data
