"""Shared full-candidate scores, immutable caches and independently checked reports."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
import csv
import multiprocessing
from pathlib import Path
import time

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from f_region_structure.pipeline import report
from wbm_retrieval.common import digest, file_hash, read_json
from wbm_retrieval.exact_metrics import order_scores
from . import scoring
from .state import KS, grid, record, seal, verified, identity, evaluation_signature

FIELDS = ['query_id', 'rank', 'candidate_id', 'similarity_score']
_WORKER = None


def read_scoring_cache(folder, feature_path=None):
    """Open already parent-verified immutable cache files inside a worker.

    Pass paths through spawn, never the mask/h512/h_views memmap objects:
    NumPy's pickle embeds their contents rather than reopening the file.
    """
    folder = Path(folder)
    required = ('map_ids', 'offsets', 'area', 'owners', 'relative_area', 'joint_all', 'histograms', 'joint')
    with np.load(folder/'regions.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in required}
    feature_path = Path(feature_path).resolve() if feature_path is not None else folder/'h512.npy'
    data.update(features=np.load(feature_path, mmap_mode='r'), folder=folder, feature_path=str(feature_path))
    return data


def _init_scores(cache_folder, feature_path, pool, matching, paths, specs):
    global _WORKER
    provider = scoring.Provider(read_scoring_cache(cache_folder, feature_path), pool, matching)
    matrices = {n: np.load(path, mmap_mode='r+') for n, path in paths.items()}
    positions = {int(s): i for i, s in enumerate(pool.gallery)}
    _WORKER = provider, matrices, positions, specs


def _score_shard(task):
    provider, matrices, positions, specs = _WORKER
    shard, rows = task
    for number, (row, q) in enumerate(rows, 1):
        parts = provider.parts(q)
        for name, spec in specs.items():
            ids, scores = provider.score(q, spec, parts)
            matrices[name][row, :] = 0.
            matrices[name][row, [positions[int(c)] for c in ids]] = scores
        if number % 50 == 0 or number == len(rows):
            print(f'Full scores shard {shard}: {number}/{len(rows)} queries, {len(specs)} variants', flush=True)
    for matrix in matrices.values():
        matrix.flush()
    return len(rows)


class CachedProvider:
    """Only reads a sealed score matrix; no new region scoring during ranking audit."""
    def __init__(self, folder, *, check=True):
        folder = Path(folder)
        meta = read_json(folder/'metadata.json')
        self.signature = meta['signature']
        self.marker = folder/'completed.json'
        if check and not verified(folder, self.signature):
            raise ValueError('Score cache is incomplete')
        with np.load(folder/'identities.npz', allow_pickle=False) as archive:
            self.query_ids = archive['queries']
            self.gallery_ids = archive['gallery']
        self.protocol_signature = meta['retrieval_protocol_signature']
        self.matrix = np.load(folder/'scores.npy', mmap_mode='r')
        self.query_positions = {int(q): i for i, q in enumerate(self.query_ids)}
        self.gallery_positions = {int(c): i for i, c in enumerate(self.gallery_ids)}
        if self.matrix.shape != (len(self.query_ids), len(self.gallery_ids)) or self.matrix.dtype != np.float64:
            raise ValueError('Score matrix dimensions or dtype differ')

    def scores(self, query, ids):
        return np.asarray(self.matrix[self.query_positions[int(query)],
                          [self.gallery_positions[int(c)] for c in ids]])

    def full_row(self, query):
        ids = self.gallery_ids[self.gallery_ids != query]
        return ids, self.scores(query, ids)


def compute_scores(output, cache, pool, matching, context_signature, specs, workers=8):
    matching = scoring.normalize_matching(matching)
    if (specs != grid()
            or type(workers) is not int or not 1 <= workers <= 8 or not len(pool)):
        raise ValueError('The full frozen grid, matching and 1–8 workers are required')
    output = Path(output)
    source_sha = file_hash(Path(cache['folder'])/'completed.json')
    feature_path = Path(cache.get('feature_path', Path(cache['folder'])/'h512.npy')).resolve()
    feature_sha = file_hash(feature_path)
    folders = {n: output/'score_cache'/pool.split_name/n for n in specs}
    paths, pending = {}, {}
    for name, spec in specs.items():
        folder = folders[name]
        signature = digest({'context': context_signature, 'split': pool.split_name,
                            'variant': name, 'spec': spec, 'score_cache_version': 1})
        if verified(folder, signature):
            meta = read_json(folder/'metadata.json')
            reused = CachedProvider(folder, check=False)
            if (meta.get('source_cache_sha256') != source_sha
                    or meta.get('h512_sha256') != feature_sha
                    or meta.get('h512_path') != str(feature_path)
                    or meta.get('spec') != spec or meta.get('variant') != name
                    or meta.get('context_signature') != context_signature
                    or reused.protocol_signature != pool.protocol_signature
                    or not np.array_equal(reused.query_ids, list(pool))
                    or not np.array_equal(reused.gallery_ids, pool.gallery)):
                raise ValueError('Existing score cache source, specification or protocol differs')
            continue
        if folder.exists():
            raise FileExistsError('Incomplete score cache requires a fresh run: '+str(folder))
        folder.mkdir(parents=True)
        matrix = np.lib.format.open_memmap(folder/'scores.npy', mode='w+', dtype=np.float64,
                   shape=(len(pool), len(pool.gallery)))
        matrix[:] = np.nan
        matrix.flush()
        del matrix
        np.savez(folder/'identities.npz', queries=np.asarray(list(pool)), gallery=np.asarray(pool.gallery))
        record(folder/'metadata.json', {'signature': signature, 'context_signature': context_signature,
               'variant': name, 'spec': spec, 'split': pool.split_name,
               'retrieval_protocol_signature': pool.protocol_signature,
               'num_queries': len(pool), 'gallery_count': len(pool.gallery),
               'dtype': 'float64', 'self_scores': 'zero; never ranked',
               'source_cache_sha256': source_sha, 'h512_path': str(feature_path), 'h512_sha256': feature_sha})
        paths[name], pending[name] = str(folder/'scores.npy'), spec
    if pending:
        rows = list(enumerate(map(int, pool)))
        partitions = [list(p) for p in np.array_split(np.asarray(rows, dtype=np.int64), min(workers, len(rows)))]
        tasks = [(i, [(int(row), int(q)) for row, q in part]) for i, part in enumerate(partitions)]
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_scores, initargs=(str(cache['folder']), str(feature_path), pool, matching, paths, pending)) as executor:
            for done in as_completed([executor.submit(_score_shard, task) for task in tasks]):
                done.result()
        for name in pending:
            matrix = np.load(paths[name], mmap_mode='r')
            for start in range(0, len(matrix), 128):
                chunk = matrix[start:start+128]
                if not np.isfinite(chunk).all() or np.any(chunk < 0) or np.any(chunk > 1):
                    raise ValueError('Incomplete or invalid full score cache')
            del matrix
            seal(folders[name], read_json(folders[name]/'metadata.json')['signature'])
        record(output/f'score_timing_{pool.split_name}.json', {'wall_seconds': time.perf_counter()-started,
               'variants': list(pending), 'workers': workers, 'candidate_pairs_per_variant': len(pool)*pool.candidates_per_query})
    if file_hash(feature_path) != feature_sha or file_hash(Path(cache['folder'])/'completed.json') != source_sha:
        raise ValueError('Source geometry or rotation features changed during scoring')
    return {name: CachedProvider(folder) for name, folder in folders.items()}


def replay_controls(output, stage, references, providers, pool):
    """Compare every saved Top500 row to its historical control, including scores."""
    from fgw_grid_search.verify import independent_order
    path = Path(output)/f'replay_{stage}.json'
    result = {'status': 'verified', 'stage': stage, 'controls': {}}
    for name, source in references.items():
        historical = Path(source['folder'])/'rankings.csv'
        if file_hash(historical) != source['ranking_sha256']:
            raise ValueError('Historical replay reference changed')
        provider, seen = providers[name], []
        if 'score_cache' in source:
            original = CachedProvider(source['score_cache'])
            if (file_hash(original.marker) != source['score_cache_completed_sha256']
                    or not np.array_equal(original.query_ids, provider.query_ids)
                    or not np.array_equal(original.gallery_ids, provider.gallery_ids)):
                raise ValueError('Original primary full-score identities changed')
            for start in range(0, len(provider.query_ids), 128):
                if not np.array_equal(original.matrix[start:start+128], provider.matrix[start:start+128]):
                    raise ValueError('Current primary full-score matrix did not replay exactly')
        with historical.open(newline='') as f:
            for q, records in groupby(csv.DictReader(f), key=lambda r: int(r['query_id'])):
                rows = list(records)
                ids = np.asarray(pool[q], dtype=np.int64)
                scores = provider.scores(q, ids)
                order = independent_order(q, ids, scores, 2026)[:max(KS)]
                if (len(rows) != len(order)
                        or [int(r['rank']) for r in rows] != list(range(1, len(order)+1))
                        or not np.array_equal(ids[order], [int(r['candidate_id']) for r in rows])
                        or not np.array_equal(scores[order], [float(r['similarity_score']) for r in rows])):
                    raise ValueError(f'Original control Top500 did not replay exactly: {stage}/{name}/{q}')
                seen.append(q)
        if not np.array_equal(seen, provider.query_ids):
            raise ValueError('Historical replay query IDs/order differs')
        result['controls'][name] = {'status': 'verified', 'source_key': source['key'],
            'source': str(historical), 'source_sha256': source['ranking_sha256'],
            'source_completed_sha256': source['completed_sha256'],
            'score_cache_sha256': file_hash(provider.marker), 'num_queries': len(seen),
            'ranking_rows': len(seen)*max(KS), 'top500_identical': True, 'max_score_error': 0.,
            'full_score_matrix_identical': 'score_cache' in source}
    record(path, result)
    return result


def _evaluate_one(task):
    output, stage, name, pool, bundle, context_signature, source_binding, reference = task
    from .verify import verify_evaluation
    output = Path(output)
    if stage == 'cross_test':
        from fgw_cross_test.state import cross_pool
        pool = cross_pool(pool, bundle.labels)
    folder = output/stage/name
    spec = grid()[name]
    signature = evaluation_signature(context_signature, stage, name)
    if not verified(folder, signature):
        if folder.exists():
            raise FileExistsError('Incomplete evaluation; preserve and inspect: '+str(folder))
        folder.mkdir(parents=True)
        provider = CachedProvider(output/'score_cache'/pool.split_name/name)
        ident = identity(output, stage, name, signature)
        cross = stage == 'cross_test'
        manifest = {**ident, 'signature': signature, 'context_signature': context_signature,
            'variant': name, 'spec': spec, 'ks': KS, 'tie_seed': 2026,
            'num_queries': len(pool), 'gallery_count': len(pool.gallery),
            'candidate_rule': 'all_gallery_except_query_and_exact_label_combination' if cross else 'all_gallery_except_query',
            'retrieval_protocol_signature': pool.protocol_signature,
            'base_protocol_signature': provider.protocol_signature,
            'provider_source': {'signature': provider.signature, 'completed_sha256': file_hash(provider.marker)},
            'source_binding_sha256': digest(source_binding),
            'ranking_scope': 'top_max_k', 'training_required': False, 'posthoc_research': True,
            'test_parameter_search': False, 'formal_protocol': True,
            'selection_rule': 'none; all six prespecified variants evaluated in every stage',
            'valid_role': 'implementation_and_protocol_gate_only',
            'frozen_plan_sha256': file_hash(output/'frozen_plan.json'),
            'replay_control': reference is not None,
            'source_evaluation_key': None if reference is None else reference['key']}
        if cross:
            manifest['derived_protocol'] = pool.metadata
        record(folder/'run_identity.json', ident)
        record(folder/'evaluation_manifest.json', manifest)
        started = time.perf_counter()
        with (folder/'rankings.csv').open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            for number, q in enumerate(pool, 1):
                ids = np.asarray(pool[q])
                scores = provider.scores(q, ids)
                order = order_scores(ids, scores, int(q), 2026)[:max(KS)]
                writer.writerows((int(q), rank, int(ids[i]), float(scores[i])) for rank, i in enumerate(order, 1))
                if number % 1500 == 0:
                    print(f'{stage}/{name} rank {number}/{len(pool)}', flush=True)
        ranking_seconds = time.perf_counter()-started
        if cross:
            coverage_metrics.evaluate_rankings_file(bundle, pool, folder/'rankings.csv', KS, folder,
                split_name=pool.split_name, metadata=manifest)
        else:
            report(folder, pool, bundle, KS, manifest)
        checked = verify_evaluation(folder, pool, bundle, KS, signature, provider, source_binding, tie_seed=2026)
        record(folder/'verification.json', checked)
        record(folder/'timing.json', {'ranking_seconds': ranking_seconds,
               'total_wall_seconds': time.perf_counter()-started,
               'verification_scope': 'independent all-query full-cache ordering and metrics; regional formula audit separate'})
        seal(folder, signature)
        print(f'Completed and verified {stage}/{name}', flush=True)
    coverage = read_json(folder/'coverage_metrics.json')
    result = {'coverage_metrics': coverage['metrics'], 'completed_sha256': file_hash(folder/'completed.json')}
    if stage != 'cross_test':
        exact = read_json(folder/'metrics.json')
        result.update(metrics=exact['metrics'], top10_hits=read_json(folder/'verification.json')['top10_hits'])
    return name, result


def evaluate(output, data, stage, context_signature, workers=8):
    from .verify import audit_formulas
    output = Path(output)
    split = 'valid' if stage == 'valid' else 'test'
    base = getattr(data['protocol'].retrieval, split)
    specs = grid()
    providers = compute_scores(output, data['cache'], base, data['config']['matching'],
                               context_signature, specs, workers)
    audit_path = output/f'formula_audit_{split}.json'
    if not audit_path.exists():
        direct = scoring.Provider(data['cache'], base, data['config']['matching'])
        audit = audit_formulas(direct, data['bundle'], specs, max_candidates=20)
        record(audit_path, audit)
    pool = base
    if stage == 'cross_test':
        from fgw_cross_test.state import cross_pool
        pool = cross_pool(base, data['bundle'].labels)
    references = data['replay_references'].get(stage, {})
    replay_controls(output, stage, references, providers, pool)
    tasks = [(str(output), stage, n, base, data['bundle'], context_signature,
              data['source_binding'], references.get(n)) for n in specs]
    results = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn')) as executor:
        for future in as_completed([executor.submit(_evaluate_one, t) for t in tasks]):
            name, result = future.result()
            results[name] = result
    results = {n: results[n] for n in specs}
    index = {'context_signature': context_signature, 'stage': stage, 'split': split,
             'specs': specs, 'results': results}
    record(output/f'evaluation_index_{stage}.json', index)
    comparison(output, stage, results)
    return index


def comparison(output, stage, results):
    """Raw proportions, overall only; full subgroup reports stay beside each ranking."""
    path = Path(output)/f'comparison_{stage}.csv'
    rows = []
    for name, result in results.items():
        row = {'variant': name, 'stage': stage}
        for kind in ('metrics', 'coverage_metrics'):
            for key, value in result.get(kind, {}).items():
                if isinstance(value, (int, float)):
                    row[kind+'.'+key] = value
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if path.exists():
        if path.read_text().splitlines() != stream.getvalue().splitlines():
            raise ValueError('Comparison table changed')
    else:
        path.write_text(stream.getvalue())
