"""Strict read-only adapter for one historical catalog-administration revision.

The two loaders below preserve the full area-fusion gates. Only the historical
code-byte lookup changes for one explicit old/current results-catalog revision.
Archived source is never executed; canonical science must still match exactly.
Legacy source_binding remains unchanged until the new input loader finishes all
historical equality checks. The receipt and its two real files are then bound.
"""
from pathlib import Path

from f_region_area_fusion import inputs as legacy
from f_region_area_fusion import scoring, selection
from f_region_area_fusion.inputs import (
    load_valid, _baseline, BASELINE, valid_inputs, sources, check_test_cache,
)
from f_region_structure.state import read_signed, signed, verified
from wbm_retrieval.common import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PATH = 'results_catalog/build.py'
HISTORICAL_SHA256 = '13d294548056ff6905968f39d0cd8f793672485b29cd8caf4674028f80f1bd6b'
CURRENT_SHA256 = 'aca87c654e2c4bf1b4576695f32376cb4272593e1b3175ce2135faedc14b65cf'
ARCHIVED_PATH = 'f_region_wbbs/provenance/results_catalog_build_13d29454.py.txt'
SCOPE = 'catalog administration only'


def resolve_source(name, expected_sha256):
    """Return the actual historical-byte path and an explicit optional receipt."""
    path = Path(name)
    if path.is_absolute() or path.as_posix() != name or '..' in path.parts:
        raise ValueError('Historical code path must be a canonical repository-relative path')
    canonical = ROOT/path
    actual = file_hash(canonical)
    if actual == expected_sha256:
        return canonical, []
    if (name != HISTORICAL_PATH or expected_sha256 != HISTORICAL_SHA256
            or actual != CURRENT_SHA256):
        raise ValueError('Historical scientific code hash differs: '+name)
    archived = ROOT/ARCHIVED_PATH
    if file_hash(archived) != HISTORICAL_SHA256:
        raise ValueError('Archived catalog source does not reproduce its exact historical bytes')
    receipt = {'historical_path': HISTORICAL_PATH, 'expected_sha256': HISTORICAL_SHA256,
        'archived_path': str(archived.resolve()), 'archived_sha256': file_hash(archived),
        'current_path': str(canonical.resolve()), 'current_sha256': actual, 'scope': SCOPE}
    return archived, [receipt]


def validate_code(code, binding=None):
    """Verify every source; optionally bind actual bytes, never fake a file hash."""
    if not isinstance(code, dict) or not code:
        raise ValueError('Historical run requires a nonempty source-code identity')
    receipts = []
    for name, expected in code.items():
        path, additions = resolve_source(name, expected)
        if binding is not None:
            valid_inputs._bind(path, binding, expected)
        receipts.extend(additions)
    receipts = unique_receipts(receipts)
    if binding is not None:
        bind_compatibility(receipts, binding)
    return receipts


def unique_receipts(receipts):
    result = {}
    for row in receipts:
        key = row['historical_path']
        if key in result and result[key] != row:
            raise ValueError('Conflicting source-compatibility receipts')
        result[key] = dict(row)
    return [result[key] for key in sorted(result)]


def bind_compatibility(receipts, binding):
    """Append truthful archive/current hashes after historical binding equalities."""
    for row in unique_receipts(receipts):
        _path, expected = resolve_source(row['historical_path'], row['expected_sha256'])
        if expected != [row]:
            raise ValueError('Source-compatibility receipt changed')
        valid_inputs._bind(Path(row['archived_path']), binding, row['archived_sha256'])
        valid_inputs._bind(Path(row['current_path']), binding, row['current_sha256'])


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
    from f_region_area_fusion.register import proposal
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
    source_compatibility = validate_code(context['code'])
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
    return manifest, wrapper, binding, source_compatibility


def load_test(source):
    source = Path(source).resolve()
    manifest, choice, binding, source_compatibility = frozen_choice(source)
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
                baseline_folder=baseline, source_compatibility=source_compatibility)
    return data
