"""Read-only frozen F/G sources and verified, already computed full WBBS scores."""
from pathlib import Path
import csv
from itertools import groupby
import numpy as np
from wbm_retrieval.common import digest, file_hash, read_json
from .cache import ids, full_candidates, validate_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'artificial/fgw_fusion_20261119T174724+0800'


def _signed(path):
    value = read_json(path)
    if value.get('signature') != digest({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Source manifest signature changed: ' + str(path))
    return value


def _bound(folder, files, name):
    relative = Path(name)
    path = (Path(folder) / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or Path(folder).resolve() not in path.parents:
        raise ValueError('Source bound file escapes its directory')
    if name not in files or file_hash(path) != files[name]:
        raise ValueError('Source file is missing or no longer bound: ' + str(path))
    return path


def _check_wbbs_origin(split, provenance, pool):
    """Check the frozen encoder/descriptor source without constructing a scorer."""
    from fgw_fusion import sources as oldsources
    from experiments.proposed.f_global_fusion.pipeline import signed_manifest, checked_files
    folder = oldsources.WBBS_RUNS[split]
    prepared = folder / split / 'prepared'
    expected_paths = {
        'manifest_sha256': folder / 'evaluation_manifest.json',
        'completed_sha256': folder / 'completed.json',
        'features_sha256': prepared / 'features.npz',
        'metadata_sha256': prepared / 'prepared_metadata.json',
        'original_code_sha256': ROOT.parent / 'compared_methods/methods/wbbs/wbbs.py'}
    if (Path(provenance['source_root']).resolve() != folder.resolve()
            or provenance.get('settings') != oldsources.WBBS_PARAMETERS
            or provenance.get('protocol_signature') != pool.protocol_signature
            or any(provenance.get(k) != file_hash(p) for k, p in expected_paths.items())):
        raise ValueError('Frozen original WBBS descriptor/code provenance changed')
    original = signed_manifest(folder / 'evaluation_manifest.json', compared=True)
    done = read_json(folder / 'completed.json')
    metadata = read_json(prepared / 'prepared_metadata.json')
    if (original.get('signature') != provenance.get('source_evaluation_signature')
            or done.get('evaluation_signature') != original['signature']
            or original.get('method') != 'wbbs' or original.get('split') != split
            or original.get('smoke') is not False
            or original.get('candidate_rule') != 'all_gallery_except_query'
            or original.get('settings', {}).get('parameters') != oldsources.WBBS_PARAMETERS
            or metadata.get('parameters') != oldsources.WBBS_PARAMETERS
            or metadata.get('protocol_signature') != pool.protocol_signature
            or metadata.get('files', {}).get('features.npz') != provenance['features_sha256']):
        raise ValueError('Frozen original WBBS evaluation/descriptor signatures differ')
    checked_files(folder, done['files'])
    return file_hash(folder / split / 'rankings.csv')


class VerifiedWbbsScores:
    """Constructed only after root, plan, matrix and previous native audit checks."""
    def __init__(self, source_run, split, pool, audit_ids, manifest, inputs, root_completed):
        self.source_run, self.split = Path(source_run), split
        self.folder = self.source_run / 'wbbs_cache' / split
        files = root_completed['files']
        prefix = 'wbbs_cache/' + split + '/'
        self.marker = _bound(self.source_run, files, prefix + 'completed.json')
        plan_path = _bound(self.source_run, files, prefix + 'plan.json')
        self.plan, self.completion = read_json(plan_path), read_json(self.marker)
        p, m = self.plan, self.completion
        self.query_ids, self.gallery_ids = ids(list(pool)).copy(), ids(pool.gallery).copy()
        self.query_ids.setflags(write=False)
        self.gallery_ids.setflags(write=False)
        self.rows = {int(q): i for i, q in enumerate(self.query_ids)}
        self.columns = {int(q): i for i, q in enumerate(self.gallery_ids)}
        self.protocol_signature = pool.protocol_signature
        self.signature = digest({'inputs': inputs['signature'], 'operation': 'full_candidate_W_cache'})
        native_ids = sorted(map(int, audit_ids))
        nq, ng = len(self.query_ids), len(self.gallery_ids)
        symmetric = set(self.query_ids) == set(self.gallery_ids)
        if (p.get('signature') != self.signature or m.get('signature') != self.signature
                or p.get('dtype') != 'float64' or p.get('backend') != 'exact_ckdtree_native_tie_fallback_v1'
                or p.get('query_ids') != self.query_ids.tolist() or p.get('gallery_ids') != self.gallery_ids.tolist()
                or p.get('symmetric_upper_triangle') is not symmetric or p.get('audit_ids') != native_ids
                or m.get('status') != 'completed' or m.get('plan_sha256') != file_hash(plan_path)
                or m.get('query_count') != nq or m.get('gallery_count') != ng
                or m.get('candidate_count') != ng - 1 or pool.candidates_per_query != ng - 1
                or m.get('represented_directed_pairs') != nq * (ng - 1)
                or m.get('scored_pairs') != (ng * (ng - 1) // 2 if symmetric else nq * (ng - 1))
                or set(m.get('row_marker_hashes', {})) != set(map(str, self.query_ids))):
            raise ValueError('Full WBBS source cache plan/identity/count differs')
        native = m.get('native_audits', {})
        if set(native) != set(map(str, native_ids)):
            raise ValueError('Missing per-class full-gallery native WBBS audit')
        for value in native.values():
            if (value.get('status') != 'verified' or value.get('pairs_checked') != ng - 1
                    or value.get('max_error') != 0 or value.get('max_abs_error') != 0
                    or value.get('scalar_equality') != 'float64 bit-for-bit'):
                raise ValueError('Native WBBS equality evidence failed')
        for q, expected in m['row_marker_hashes'].items():
            if file_hash(self.folder / 'rows' / q / 'completed.json') != expected:
                raise ValueError('Source WBBS sealed row marker changed: ' + q)
        matrix_path = self.folder / 'scores.npy'
        if matrix_path.is_symlink() or file_hash(matrix_path) != m.get('scores_sha256'):
            raise ValueError('Source WBBS full matrix hash changed')
        self.matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
        validate_matrix(self.matrix, self.query_ids, self.gallery_ids)
        audit_path = _bound(self.source_run, files, f'source_audit_{split}.json')
        self.source_audit = _signed(audit_path)
        a = self.source_audit
        replay = a.get('wbbs_all_query_source_replay', {})
        fg = a.get('fg_full_gallery', {})
        if (a.get('status') != 'verified' or a.get('split') != split
                or a.get('input_context_signature') != digest(manifest['context'])
                or a.get('wbbs_cache_completed_sha256') != file_hash(self.marker)
                or a.get('wbbs_native_full_gallery') != native
                or replay.get('status') != 'verified' or replay.get('num_queries') != nq
                or replay.get('candidates_per_query') != ng - 1 or replay.get('top_k') != 500
                or replay.get('max_score_error') != 0
                or a.get('wbbs_raw_descriptor_replay', {}).get('status') != 'verified'
                or a.get('wbbs_raw_descriptor_replay', {}).get('query_ids') != native_ids
                or fg.get('status') != 'verified' or fg.get('query_count') != len(native_ids)
                or sorted(x['query_id'] for x in fg.get('queries', [])) != native_ids
                or any(x.get('candidate_count') != ng - 1 or x.get('all_three_top500_exact') is not True
                       for x in fg.get('queries', []))):
            raise ValueError('Signed frozen source audit chain differs')
        independent_path = _bound(self.source_run, files, f'independent_cache_audit/{split}.json')
        independent = read_json(independent_path)
        expected = {'status': 'verified', 'split': split, 'matrix_shape': [nq, ng], 'matrix_dtype': 'float64',
                    'matrix_sha256': m['scores_sha256'], 'cache_completed_sha256': file_hash(self.marker),
                    'source_audit_sha256': file_hash(audit_path), 'source_audit_signature': a['signature'],
                    'source_context_signature': digest(manifest['context']),
                    'frozen_query_ids_order_match': True, 'frozen_gallery_ids_order_match': True,
                    'row_markers_verified': nq, 'row_npz_verified': nq, 'row_to_matrix_bitwise_equal': True,
                    'candidate_pairs_verified': nq * (ng - 1), 'self_cells_nan': nq,
                    'native_audit_queries': len(native_ids), 'native_audit_pairs': len(native_ids) * (ng - 1),
                    'native_audit_max_error': 0, 'old_WBBS_top500_queries_replay_evidence': nq,
                    'old_WBBS_top500_source_scores_max_error': 0,
                    'old_WBBS_ranking_sha256': replay.get('ranking_sha256'),
                    'raw_descriptor_audit_queries': len(native_ids), 'independent_FG_audit_queries': len(native_ids),
                    'three_fusion_top500_all_exact': True}
        if any(independent.get(k) != v for k, v in expected.items()):
            raise ValueError('Independent completed full-cache audit evidence differs')
        self.ranking_path = self.source_run / split / 'FGW/rankings.csv'
        eval_marker = _bound(self.source_run, files, f'{split}/FGW/completed.json')
        self.ranking_path = _bound(eval_marker.parent, read_json(eval_marker)['files'], 'rankings.csv')
        self.provenance = {'source_root': str(self.source_run.resolve()), 'split': split,
            'root_signature': manifest['signature'], 'root_completed_sha256': file_hash(self.source_run / 'completed.json'),
            'input_signature': inputs['signature'], 'input_sha256': file_hash(self.source_run / f'inputs_{split}.json'),
            'protocol_signature': pool.protocol_signature, 'cache_signature': self.signature,
            'cache_plan_sha256': file_hash(plan_path), 'cache_completed_sha256': file_hash(self.marker),
            'scores_sha256': m['scores_sha256'], 'source_audit_sha256': file_hash(audit_path),
            'source_audit_signature': a['signature'], 'independent_cache_audit_sha256': file_hash(independent_path),
            'source_fgw_ranking_sha256': file_hash(self.ranking_path),
            'native_audit_queries': len(native_ids), 'native_audit_max_error': 0,
            'wbbs_recomputed': False}

    def scores(self, query, candidate_ids):
        q, c = full_candidates(query, candidate_ids, self.gallery_ids, self.rows)
        return np.asarray(self.matrix[self.rows[q], [self.columns[int(x)] for x in c]])


def load_inputs(source_run=DEFAULT_SOURCE, split='valid'):
    """Load frozen F/G and only the requested W split; never train or evaluate."""
    if split not in ('valid', 'test'):
        raise ValueError('Only explicit valid/test input splits are supported')
    from fgw_fusion import register, sources as oldsources
    from b1_ablation import run as oldrun
    from experiments.proposed.f_global_fusion.pipeline import checked_files
    source_run = Path(source_run).resolve()
    register.proposal(source_run)
    manifest = _signed(source_run / 'run_manifest.json')
    context = manifest['context']
    checked_files(ROOT, context['code'])
    if (Path(context['plan']['source_root']).resolve() != oldsources.SOURCE.resolve()
            or Path(context['plan']['reference_root']).resolve() != oldsources.REFERENCE.resolve()
            or context['reference_manifest_sha256'] != file_hash(oldsources.REFERENCE / 'run_manifest.json')):
        raise ValueError('Frozen F/G source or reference identity differs')
    inputs = _signed(source_run / f'inputs_{split}.json')
    if (inputs.get('parent_signature') != manifest['signature'] or inputs.get('split') != split
            or inputs.get('provenance') != context['inputs'][split] or inputs.get('code') != context['code']):
        raise ValueError('Frozen source input/root binding differs')
    bundle, contexts = oldrun.load_inputs(oldsources.SOURCE, 'cuda')
    pool, scorer, _, _, current = contexts[split]
    expected_counts = (500, 3599) if split == 'valid' else (7199, 7199)
    if (len(pool), len(pool.gallery)) != expected_counts or pool.candidates_per_query != expected_counts[1] - 1:
        raise ValueError('Search requires the formal deduplicated complete candidate protocol')
    recorded = {k: v for k, v in inputs['provenance'].items() if k != 'wbbs'}
    if current != recorded:
        raise ValueError('Frozen F/G provenance changed since completed FGW')
    original_wbbs_ranking = _check_wbbs_origin(split, inputs['provenance']['wbbs'], pool)
    audit_ids = oldsources.selected_queries(pool, bundle.labels)
    if len(audit_ids) != 37:
        raise ValueError('The 37 frozen complete-label combinations are required')
    provider = VerifiedWbbsScores(source_run, split, pool, audit_ids, manifest, inputs,
                                  read_json(source_run / 'completed.json'))
    if provider.source_audit['wbbs_all_query_source_replay']['ranking_sha256'] != original_wbbs_ranking:
        raise ValueError('WBBS replay evidence does not bind the original frozen ranking')
    provenance = {**provider.provenance, 'fg': current, 'source_code': context['code'],
                  'source_manifest_sha256': file_hash(source_run / 'run_manifest.json')}
    return bundle, pool, scorer, provider, provenance


def audit_components(pool, bundle, scorer, provider):
    """Independently audit F/G and old FGW Top500; reuse bound native W evidence."""
    from fgw_fusion.sources import independent_fg, selected_queries
    from b1_ablation.verify import independent_order, SOURCE_ATOL
    if not isinstance(provider, VerifiedWbbsScores) or provider.split != pool.split_name:
        raise ValueError('A verified complete WBBS source provider is required')
    chosen = selected_queries(pool, bundle.labels)
    if chosen != sorted(map(int, provider.completion['native_audits'])) or len(chosen) != 37:
        raise ValueError('Per-class audit query identity differs')
    selected = set(chosen)
    audits = []
    with provider.ranking_path.open(newline='') as handle:
        groups = groupby(csv.DictReader(handle), key=lambda r: int(r['query_id']))
        for q_ in pool:
            q = int(q_)
            item = next(groups, None)
            if item is None or item[0] != q:
                raise ValueError('Source FGW query identity/order changed')
            rows = list(item[1])
            if len(rows) != 500 or [int(r['rank']) for r in rows] != list(range(1, 501)):
                raise ValueError('Source FGW must retain exactly ranks 1..500 per query')
            if q not in selected:
                continue
            c = np.asarray(pool[q], dtype=np.int64)
            f, g = independent_fg(q, c, scorer)
            actual = scorer.components(q, c)
            af = np.asarray(actual['optimal_f_score'], dtype=np.float64)
            ag = np.asarray(actual['global_score'], dtype=np.float64)
            np.testing.assert_allclose(f, af, atol=SOURCE_ATOL, rtol=SOURCE_ATOL)
            np.testing.assert_allclose(g, ag, atol=SOURCE_ATOL, rtol=SOURCE_ATOL)
            w = provider.scores(q, c)
            direct = provider.matrix[provider.rows[q], [provider.columns[int(x)] for x in c]]
            if not np.array_equal(w, direct):
                raise ValueError('WBBS provider differs from its bound source matrix')
            recomputed, independent = (.5 * af + .5 * ag) * w, (.5 * f + .5 * g) * w
            order, independent_ix = independent_order(q, c, recomputed)[:500], independent_order(q, c, independent)[:500]
            ranked = np.array([int(r['candidate_id']) for r in rows], dtype=np.int64)
            saved = np.array([float(r['similarity_score']) for r in rows], dtype=np.float64)
            if not np.array_equal(c[order], ranked) or not np.array_equal(c[independent_ix], ranked):
                raise ValueError('Independent FGW source Top500 ordering differs: ' + str(q))
            if not np.array_equal(recomputed[order], saved):
                raise ValueError('Frozen FGW saved score differs: ' + str(q))
            for field, vector in [('optimal_f_score', af), ('global_score', ag), ('wbbs_score', w)]:
                if not np.array_equal(vector[order], np.array([float(r[field]) for r in rows])):
                    raise ValueError('Source FGW component score differs: ' + field)
            audits.append({'query_id': q, 'candidate_count': len(c), 'F_max_error': float(np.max(np.abs(f - af))),
                           'G_max_error': float(np.max(np.abs(g - ag))), 'W_max_error': 0.,
                           'FGW_max_error': 0., 'FGW_top500_exact': True})
        if next(groups, None) is not None:
            raise ValueError('Extra source FGW query')
    return {'status': 'verified', 'split': pool.split_name, 'query_count': len(audits), 'queries': audits,
            'selection': 'minimum original query ID in each of 37 full-label combinations',
            'source_fgw_ranking_sha256': file_hash(provider.ranking_path),
            'wbbs_native_audit_reused': provider.provenance,
            'wbbs_recomputed': False, 'independent_FG_tolerance': SOURCE_ATOL}


def inventory(source_run=DEFAULT_SOURCE, provenance=None):
    """Inventory old inputs and bound results; never walk new-run shard trees."""
    from fgw_fusion import sources as oldsources
    from b1_ablation import run as oldrun
    source_run = Path(source_run).resolve()
    result = oldsources.inventory()
    manifest = _signed(source_run / 'run_manifest.json')
    files = {source_run / 'completed.json', source_run / 'run_manifest.json'}
    root_done = read_json(source_run / 'completed.json')
    for name in root_done['files']:
        files.add(_bound(source_run, root_done['files'], name))
    files.update(ROOT / name for name in manifest['context']['code'])
    for split in ('valid', 'test'):
        for name in ('FGW', 'GW', 'FW'):
            folder = source_run / split / name
            marker = read_json(folder / 'completed.json')
            files.update(_bound(folder, marker['files'], rel) for rel in marker['files'])
        folder = source_run / 'analysis' / split
        marker = read_json(folder / 'completed.json')
        files.update(_bound(folder, marker['files'], rel) for rel in marker['files'])
        cache = source_run / 'wbbs_cache' / split
        files.add(cache / 'scores.npy')
        marker = read_json(cache / 'completed.json')
        files.update(cache / 'rows' / q / 'completed.json' for q in marker['row_marker_hashes'])
    # load_inputs checks both frozen F/G contexts, so protect both input caches.
    defaults = oldrun.oldrun.parser().parse_args([])
    original = _signed(oldsources.SOURCE / 'run_manifest.json')
    training = oldrun.path(defaults.source_root) / 'cc8'
    files.update((training / name).resolve() for name in original['context']['geometry']['source_files'])
    runtime = read_json(training / 'seed_2026/F/run_manifest.json')['config']
    files.add(Path(runtime['data_file']).resolve())
    for key in ('split_manifest', 'deduplication_metadata'):
        if Path(runtime[key]).exists():
            files.add(Path(runtime[key]).resolve())
    if Path(runtime['retrieval_protocol_dir']).exists():
        files.update(p for p in Path(runtime['retrieval_protocol_dir']).rglob('*') if p.is_file())
    for split in ('valid', 'test'):
        local = oldrun.oldrun.frozen_local_cache(defaults, split, digest(original['context']['geometry']))
        global_root = oldrun.path(getattr(defaults, 'wapirl_' + split + '_root'))
        for folder in (local, global_root):
            files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    if provenance is not None and provenance.get('root_signature') != manifest['signature']:
        raise ValueError('Inventory source differs from loaded provenance')
    for p in sorted(files):
        stat = p.stat()
        result[str(p.resolve())] = {'sha256': file_hash(p), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return result
