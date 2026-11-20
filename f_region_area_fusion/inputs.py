"""Read-only input reuse; test is inaccessible until all 29 valid runs seal."""
from pathlib import Path

from f_region_score_search import inputs as valid_inputs
from f_region_score_search import scoring as previous_scoring
from f_region_score_test.inputs import _check_cache as check_test_cache
from f_region_structure import sources
from wbm_retrieval.common import digest, file_hash, read_json
from . import scoring, selection
from .state import read_signed, signed, verified

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'artificial/f_region_structure_20261120T120331+0800'
BASELINE = 'mul_a0p5_b0p15'
OLD_NAME = 'mixed_a0p5_e1p0'
OLD_RUNS = {
    'valid': ROOT / 'artificial/f_region_score_search_20261120T140544+0800',
    'test': ROOT / 'artificial/f_region_score_test_20261120T143348+0800',
}


def _baseline(split, binding):
    """Bind only the historical equal-weight result used for exact replay."""
    root = OLD_RUNS[split]
    valid_inputs._bind(root/'run_manifest.json', binding)
    manifest = read_signed(root/'run_manifest.json')
    spec = previous_scoring.frozen_grid()[OLD_NAME]
    signature = digest({'context': digest(manifest['context']), 'split': split,
                        'variant': OLD_NAME, 'spec': spec})
    folder = root/split/OLD_NAME
    valid_inputs._sealed(folder, signature, valid_inputs.EVALUATION_FILES, binding)
    meta = read_json(folder/'evaluation_manifest.json')
    check = read_json(folder/'verification.json')
    q = 500 if split == 'valid' else 7199
    if (meta.get('spec') != spec or meta.get('split') != split or meta.get('variant') != OLD_NAME
            or check.get('status') != 'verified' or check.get('num_queries') != q
            or check.get('ranking_rows') != q*500
            or check.get('all_queries_full_candidate_topk_exact') is not True
            or check.get('ranking_sha256') != file_hash(folder/'rankings.csv')):
        raise ValueError('Historical baseline must be a complete verified same-protocol evaluation')
    return folder


def load_valid(source=DEFAULT_SOURCE):
    source = Path(source).resolve()
    data = valid_inputs.load(source)
    # Preserve the exact upstream binding separately: test must reproduce it.
    data['cache_source_binding'] = dict(data['source_binding'])
    data['baseline_folder'] = _baseline('valid', data['source_binding'])
    data['source_binding'] = dict(sorted(data['source_binding'].items()))
    data.update(source_run=source, cache_source_run=source)
    return data


def frozen_choice(source):
    source = Path(source).resolve()
    manifest = read_signed(source/'run_manifest.json')
    context = manifest['context']
    signature = digest(context)
    if (manifest.get('version') != 'f_region_area_fusion_v1' or context.get('split') != 'valid'
            or context.get('grid') != scoring.frozen_grid()
            or context.get('full_grid') != scoring.frozen_grid()
            or context.get('selection_rule') != selection.RULE or not verified(source, signature)):
        raise ValueError('Test requires the sealed complete 29-candidate valid run')
    # Reuse the full read-only registration validator before any test input is
    # opened. It cross-checks every metric, ranking hash and verification file.
    from .register import proposal
    proposed = proposal(source)
    if len(proposed['runs']) != 29 or any(row['split'] != 'valid' for row in proposed['runs']):
        raise ValueError('Every valid result must pass complete formal validation')
    index = read_json(source/'evaluation_index_valid.json')
    choice = read_signed(source/'selection.json')
    expected = selection.select_top10(index['results'])
    if (index.get('context_signature') != signature or index.get('split') != 'valid'
            or index.get('specs') != scoring.frozen_grid()
            or choice.get('context_signature') != signature
            or choice.get('valid_index_sha256') != file_hash(source/'evaluation_index_valid.json')
            or any(choice.get(k) != v for k, v in expected.items())):
        raise ValueError('Frozen Top-10 differs from the full valid index and selection rule')
    for name, spec in scoring.frozen_grid().items():
        folder = source/'valid'/name
        esig = digest({'context': signature, 'split': 'valid', 'variant': name, 'spec': spec})
        if not verified(folder, esig):
            raise ValueError('Every valid candidate must be completed')
        evidence = read_json(folder/'verification.json')
        row = index['results'][name]
        if (evidence.get('status') != 'verified' or evidence.get('signature') != esig
                or evidence.get('num_queries') != 500 or evidence.get('ranking_rows') != 250000
                or evidence.get('full_candidates_checked') != 1799000
                or evidence.get('all_queries_full_candidate_topk_exact') is not True
                or evidence.get('top10_hits') != row['top10_hits']
                or row['completed_sha256'] != file_hash(folder/'completed.json')
                or row['metrics'] != read_json(folder/'metrics.json')['metrics']
                or row['coverage_metrics'] != read_json(folder/'coverage_metrics.json')['metrics']):
            raise ValueError('Valid selection index differs from its completed evidence')
    for name, expected_hash in context['code'].items():
        if file_hash(ROOT/name) != expected_hash:
            raise ValueError('Scientific code changed since valid: '+name)
    wrapper = signed({'schema_version': 1, 'source_run': str(source),
        'source_selection_sha256': file_hash(source/'selection.json'),
        'source_completed_sha256': file_hash(source/'completed.json'),
        'source_valid_index_sha256': file_hash(source/'evaluation_index_valid.json'),
        'selection': choice, 'selected_variants': choice['selected_variants'],
        'selected_specs': choice['selected_specs'], 'selection_policy': 'frozen_valid_top10_only'})
    binding = {}
    valid_inputs._bind(source/'completed.json', binding)
    for name, expected_hash in read_json(source/'completed.json')['files'].items():
        valid_inputs._bind(source/name, binding, expected_hash)
    return manifest, wrapper, binding


def load_test(source):
    source = Path(source).resolve()
    manifest, choice, binding = frozen_choice(source)
    old_run = Path(manifest['source_run']).resolve()
    data = load_valid(old_run)
    if (data['source_binding'] != manifest['context']['source_binding']
            or data['config']['matching'] != manifest['context']['matching']):
        raise ValueError('Test must reuse the exact model, cache source, protocol and matcher bound by valid')
    for name, expected in data['source_binding'].items():
        valid_inputs._bind(Path(name), binding, expected)
    old_manifest = read_signed(old_run/'run_manifest.json')
    folder = old_run/'cache/test'
    signature = digest({'context': data['old_context_signature'], 'split': 'test', 'cache_version': 1})
    valid_inputs._sealed(folder, signature, valid_inputs.CACHE_FILES, binding)
    if (folder/'h_views.npy').exists():
        raise ValueError('Test cannot contain unbound training views')
    upstream = read_json(folder/'source_binding.json')
    roots = [(Path(p).resolve()/'test') for p in old_manifest['feature_roots']]
    if not isinstance(upstream, dict) or not upstream:
        raise ValueError('Test cache lacks upstream bindings')
    for name, expected in upstream.items():
        path = Path(name)
        if not path.is_absolute() or not any(path.resolve().is_relative_to(root) for root in roots):
            raise ValueError('Test cache dependency escaped its declared source')
        valid_inputs._bind(path, binding, expected)
    cache = sources.read_cache(folder)
    check_test_cache(cache, data['protocol'])
    audit = read_json(folder/'audit.json')
    if (audit.get('split') != 'test' or audit.get('maps') != 7199
            or audit.get('regions') != int(cache['offsets'][-1])
            or audit.get('map_ids') != cache['map_ids'].tolist()):
        raise ValueError('Test descriptor audit and cache identities differ')
    baseline = _baseline('test', binding) if BASELINE in choice['selected_specs'] else None
    data.update(cache=cache, source_binding=dict(sorted(binding.items())),
                frozen_selection=choice, source_run=source, cache_source_run=old_run,
                baseline_folder=baseline)
    return data
