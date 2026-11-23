"""Frozen regional CNN-only/fusion evaluation, without changing historical code.

Every model scores the full ordinary gallery once. Ordinary and cross-combination
rankings are independently produced from those complete float64 score rows. The
existing independent metric verifiers audit every query, full-pool denominator
and saved Top-K; regional formulas have a separate deterministic sampled audit.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
from itertools import groupby
import multiprocessing
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from f_region_area_fusion.scoring import score_components as historical_fusion
from f_region_evidence.scoring import RegionPairScorer, cache_layout
from f_region_evidence.verify import independent_components, independent_greedy
from f_region_structure.pipeline import report
from f_region_structure import verify as ordinary_verify
from f_region_structure.state import now, record, seal, verified
from f_region_wbbs.protocol import KS, PROTOCOL, RULE, audit_pool, make_cross_pool
from f_region_wbbs.verify import _verify_cross
from fgw_grid_search.verify import independent_order
from paper_c_ablation.pipeline import CachedProvider, read_scoring_cache
from paper_c_ablation.scoring import normalize_matching
from wbm_retrieval.common import digest, file_hash, read_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import order_scores

MODELS = ('baseline', 'context_dropout', 'filtered_negatives')
STYLES = ('cnn_only', 'fusion')
FIELDS = ('query_id', 'rank', 'candidate_id', 'similarity_score')
FUSION = {'family': 'add', 'alpha': .4, 'beta': .25}


def code_identity():
    """Bind this evaluator and the production/independent paths it executes."""
    project=Path(__file__).resolve().parents[1]
    names=('cnn_refinement/evaluation.py',
        'paper_c_ablation/pipeline.py','paper_c_ablation/scoring.py',
        'f_region_area_fusion/scoring.py',
        'f_region_evidence/scoring.py','f_region_evidence/verify.py',
        'f_region_structure/pipeline.py','f_region_structure/scoring.py',
        'f_region_structure/verify.py','f_region_structure/state.py',
        'f_region_wbbs/protocol.py','f_region_wbbs/verify.py',
        'fgw_grid_search/verify.py','fgw_cross_test/verify.py','fgj_cross_test/verify.py',
        'wbm_retrieval/common.py','wbm_retrieval/diagnostics.py','wbm_retrieval/exact_metrics.py',
        'experiments/proposed/f_segmentation_geometry/coverage_metrics.py',
        'experiments/proposed/f_segmentation_geometry/scoring.py')
    paths=[project/name for name in names]
    paths.append(project.parent/'compared_methods/shared/cross_combination.py')
    return {str(path.resolve()):file_hash(path) for path in paths}


def specifications():
    return {name: {'region_pair': 'clip(cos(h_q,h_c),0,1)' if name == 'cnn_only'
                  else '0.75*(0.4*C+0.6*T)+0.25*A',
                  'matching': 'original_greedy_one_to_one', 'weight': 'query_sqrt_area',
                  'top_regions': 4, 'structure': None if name == 'cnn_only' else 'frozen_original_J',
                  'geometry': None if name == 'cnn_only' else 'frozen_original_relative_area'}
            for name in STYLES}


def token_layout(cache):
    """Portable row-order identity; row count alone cannot identify region tokens."""
    cache_layout(cache, max_regions=4)
    return {key: {'dtype': str(np.asarray(cache[key]).dtype),
                  'shape': list(np.asarray(cache[key]).shape),
                  'sha256': hashlib.sha256(np.asarray(cache[key]).tobytes()).hexdigest()}
            for key in ('map_ids', 'offsets', 'owners', 'area')}


def replaced_cache(source, h512_path):
    """Shallow read-only view retaining all source geometry/descriptor arrays."""
    features = np.load(h512_path, mmap_mode='r', allow_pickle=False)
    if features.shape != (len(source['features']), 512) or features.dtype not in (np.float32,np.float64):
        raise ValueError('h512 must retain source token count, floating dtype and width 512')
    result = dict(source)
    result['features'] = features
    # Validates finite normalized-or-zero rows without silently changing bytes.
    RegionPairScorer(result)
    cache_layout(result, max_regions=4)
    return result


class Provider:
    """Reuse only the frozen channel arithmetic and original region matcher."""
    def __init__(self, cache, pool, matching=None):
        self.cache, self.pool = cache, pool
        self.matching = normalize_matching(matching)
        ids, self.offsets, self.area = cache_layout(cache, max_regions=4)
        gallery = np.asarray(pool.gallery, dtype=np.int64)
        if (set(ids.tolist()) != set(gallery.tolist()) or len(gallery) != len(ids)
                or pool.candidates_per_query != len(gallery)-1):
            raise ValueError('Scoring must use all gallery maps excluding only self')
        self.positions = {int(q): i for i, q in enumerate(ids)}
        self.padded = np.full((len(ids), int(np.max(np.diff(self.offsets), initial=0))),
                              -1, dtype=np.int64)
        for i, (a, b) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            self.padded[i, :b-a] = np.arange(a, b)
        self.engine = RegionPairScorer(cache)

    def tokens(self, query):
        i = self.positions[int(query)]
        return np.arange(self.offsets[i], self.offsets[i+1], dtype=np.int64)

    def parts(self, query):
        return self.engine.components(self.tokens(query))

    @staticmethod
    def pair_scores(parts, style):
        if style == 'cnn_only':
            return parts['cosine']
        if style == 'fusion':
            return historical_fusion(parts, FUSION)
        raise ValueError('Unknown fixed CNN evaluation style')

    def score(self, query, style, parts=None):
        ids = np.asarray(self.pool[int(query)], dtype=np.int64)
        expected = np.asarray(self.pool.gallery)[np.asarray(self.pool.gallery) != query]
        if not np.array_equal(ids, expected):
            raise ValueError('Scoring candidate identities/order changed')
        qt = self.tokens(query)
        parts = self.parts(query) if parts is None else parts
        pair = self.pair_scores(parts, style)
        rows = self.padded[[self.positions[int(c)] for c in ids]]
        scores = batched_map_scores(pair, rows, self.area[qt], self.matching)
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError('Invalid map-level CNN score')
        return ids, scores


def audit_formulas(provider, bundle, max_candidates=20):
    representatives = {}
    for q in sorted(map(int, provider.pool)):
        representatives.setdefault(int(bundle.class_ids[q]), q)
    details = {name: {'candidate_pairs': 0, 'max_raw_formula_absolute_error': 0.,
                     'max_matching_absolute_error': 0.} for name in STYLES}
    chosen = {}
    for q in representatives.values():
        ids = np.sort(np.asarray(provider.pool[q]))[:max_candidates]
        chosen[str(q)] = ids.tolist()
        qt = provider.tokens(q)
        tokens = [provider.tokens(c) for c in ids]
        ct = np.concatenate(tokens)
        direct = independent_components(provider.cache, qt, ct)
        parts = provider.parts(q)
        for style in STYLES:
            pair = provider.pair_scores(parts, style)[:, ct]
            expected = direct['cosine' if style == 'cnn_only' else 'raw_score']
            error = float(np.max(np.abs(pair-expected), initial=0.))
            if error > 2e-6:
                raise ValueError('Independent regional formula mismatch: '+style)
            allids, scores = provider.score(q, style, parts)
            lookup = dict(zip(map(int, allids), map(float, scores)))
            details[style]['max_raw_formula_absolute_error'] = max(
                details[style]['max_raw_formula_absolute_error'], error)
            start = 0
            for candidate, rows in zip(ids, tokens):
                # Match the checked production pair matrix, so float32 cosine
                # near-ties cannot masquerade as a greedy implementation error.
                scalar = independent_greedy(pair[:, start:start+len(rows)], provider.area[qt])
                start += len(rows)
                delta = abs(scalar-lookup[int(candidate)])
                if delta > 2e-6:
                    raise ValueError('Independent greedy aggregation mismatch')
                detail = details[style]
                detail['candidate_pairs'] += 1
                detail['max_matching_absolute_error'] = max(detail['max_matching_absolute_error'], delta)
    return {'status': 'verified', 'num_classes': len(representatives),
            'query_ids': list(representatives.values()), 'candidate_ids_by_query': chosen,
            'variants': details, 'formula_atol': 2e-6,
            'scope': 'Each class minimum query ID, first 20 sorted candidates: separate scalar C/H/T/A arithmetic, followed by separate scalar original greedy matching on the checked production pair matrix. This is a sampled formula audit.'}


_WORKER = None


def _init_scores(cache_folder, h512_path, pool, matching, paths):
    global _WORKER
    cache = replaced_cache(read_scoring_cache(cache_folder), h512_path)
    matrices = {name: np.load(path, mmap_mode='r+') for name, path in paths.items()}
    _WORKER = (Provider(cache, pool, matching), matrices,
               {int(c): i for i, c in enumerate(pool.gallery)})


def _score_shard(task):
    shard, rows = task
    provider, matrices, positions = _WORKER
    for number, (row, query) in enumerate(rows, 1):
        parts = provider.parts(query)
        for name, matrix in matrices.items():
            ids, values = provider.score(query, name, parts)
            matrix[row, positions[query]] = 0.
            matrix[row, [positions[int(c)] for c in ids]] = values
        if number % 250 == 0:
            print(f'CNN full-score shard {shard}: {number}/{len(rows)}', flush=True)
    for matrix in matrices.values():
        matrix.flush()
    return len(rows)


def _preserve_incomplete(folder, output):
    if folder.exists():
        archive = Path(output)/'interrupted_evaluations'
        archive.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix=folder.name+'_', dir=archive))/'partial'
        folder.rename(destination)


def compute_scores(output, cache, pool, matching, h512_path, model_context, workers=4):
    output, h512_path = Path(output), Path(h512_path).resolve()
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Use 1 to 16 evaluation workers')
    specs = specifications()
    source_sha = file_hash(Path(cache['folder'])/'completed.json')
    feature_sha = file_hash(h512_path)
    layout = token_layout(cache)
    paths, folders = {}, {}
    for style in STYLES:
        folder = output/'score_cache'/pool.split_name/style
        folders[style] = folder
        science = {'context_signature': model_context, 'variant': style, 'spec': specs[style],
            'split': pool.split_name, 'retrieval_protocol_signature': pool.protocol_signature,
            'source_cache_sha256': source_sha, 'h512_sha256': feature_sha,
            'token_layout': layout, 'num_queries': len(pool), 'gallery_count': len(pool.gallery),
            'dtype': 'float64', 'self_scores': 'zero; never ranked', 'cache_version': 1}
        signature = digest(science)
        metadata = {**science, 'signature': signature}
        if verified(folder, signature):
            if read_json(folder/'metadata.json') != metadata:
                raise ValueError('Completed score-cache metadata changed')
            old = CachedProvider(folder, check=False)
            if not np.array_equal(old.query_ids, list(pool)) or not np.array_equal(old.gallery_ids, pool.gallery):
                raise ValueError('Score-cache identities changed')
            continue
        _preserve_incomplete(folder, output)
        folder.mkdir(parents=True)
        matrix = np.lib.format.open_memmap(folder/'scores.npy', mode='w+', dtype=np.float64,
                                          shape=(len(pool), len(pool.gallery)))
        matrix[:] = np.nan
        matrix.flush()
        del matrix
        np.savez(folder/'identities.npz', queries=np.asarray(list(pool), dtype=np.int64),
                 gallery=np.asarray(pool.gallery, dtype=np.int64))
        record(folder/'metadata.json', metadata)
        paths[style] = str(folder/'scores.npy')
    if paths:
        tasks = [(i, [(int(row), int(q)) for row, q in part])
                 for i, part in enumerate(np.array_split(np.array(list(enumerate(pool))), min(workers, len(pool))))]
        args = (str(cache['folder']), str(h512_path), pool, matching, paths)
        if workers == 1:
            _init_scores(*args)
            for task in tasks:
                _score_shard(task)
        else:
            with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init_scores, initargs=args) as executor:
                for future in as_completed([executor.submit(_score_shard, task) for task in tasks]):
                    future.result()
        for style in paths:
            matrix = np.load(paths[style], mmap_mode='r')
            for start in range(0, len(matrix), 128):
                rows = matrix[start:start+128]
                if not np.isfinite(rows).all() or np.any(rows < 0) or np.any(rows > 1):
                    raise ValueError('Incomplete full-gallery score matrix')
            seal(folders[style], read_json(folders[style]/'metadata.json')['signature'])
    if feature_sha != file_hash(h512_path) or source_sha != file_hash(Path(cache['folder'])/'completed.json'):
        raise ValueError('Feature/source cache changed during scoring')
    return {style: CachedProvider(folder) for style, folder in folders.items()}


def _identity(output, stage, style, signature):
    path = Path(output)/'identities'/f'{stage}__{style}.json'
    key = 'cnn_refinement_'+re.sub(r'[^A-Za-z0-9_]+', '_', str(Path(output).parent.name+'_'+Path(output).name))+'_'+stage+'_'+style
    science = {'schema_version': 1, 'key': key, 'evaluation_signature': signature,
               'variant': style, 'split': stage.removeprefix('cross_'), 'stage': stage,
               'protocol': PROTOCOL if stage.startswith('cross_') else 'retrieval_v1'}
    if path.exists():
        old = read_json(path)
        stamp = datetime.fromisoformat(old['started_at'])
        if (any(old.get(k) != v for k, v in science.items())
                or old['run_id'] != stamp.strftime('%Y%m%dT%H%M%S%z')+'__'+key
                or not old['started_at'].endswith('+08:00')):
            raise ValueError('Persisted evaluation identity changed')
        return old
    stamp = now()
    identity = {**science, 'started_at': stamp.isoformat(),
                'run_id': stamp.strftime('%Y%m%dT%H%M%S%z')+'__'+key}
    record(path, identity)
    return identity


def verify_evaluation(folder, pool, bundle, provider, signature, source_binding, *, formal=True, ks=KS):
    """All scores/rankings/metrics independently audited, with split-aware cross N."""
    folder = Path(folder)
    manifest = read_json(folder/'evaluation_manifest.json')
    identity = read_json(folder/'run_identity.json')
    stage, style = manifest['stage'], manifest['variant']
    cross = stage.startswith('cross_')
    audit = audit_pool(pool, bundle.labels, ks, formal=formal)
    cache_binding = {'signature': provider.signature, 'completed_sha256': file_hash(provider.marker)}
    expected_signature = digest({'context': manifest['context_signature'], 'stage': stage,
                                  'variant': style, 'spec': specifications()[style]})
    if (manifest['signature'] != signature or signature != expected_signature
            or manifest['spec'] != specifications()[style] or stage not in ('valid','test','cross_valid','cross_test')
            or manifest['split'] != pool.split_name or manifest['ks'] != list(ks)
            or manifest['tie_seed'] != 2026 or manifest['provider_source'] != cache_binding
            or manifest['source_binding_sha256'] != digest(source_binding)
            or not np.array_equal(provider.query_ids, list(pool)) or not np.array_equal(provider.gallery_ids, pool.gallery)
            or any(manifest.get(k) != audit[k] for k in ('protocol','retrieval_protocol_signature','candidate_rule','num_queries','gallery_count'))
            or identity['evaluation_signature'] != signature
            or any(manifest.get(k) != v for k,v in identity.items())):
        raise ValueError('Evaluation identity, cache, model, source or protocol mismatch')
    if cross and (manifest.get('derived_protocol') != pool.metadata
                  or any((folder/name).exists() for name in ('metrics.json','per_query.csv','per_class.csv'))):
        raise ValueError('Cross evaluation cannot emit exact-combination metrics')
    families = ['coverage_metrics.json'] if cross else ['metrics.json','coverage_metrics.json']
    for name in families:
        metadata = read_json(folder/name)['metadata']
        if any(metadata.get(k) != v for k,v in manifest.items()):
            raise ValueError('Report detached from immutable manifest')
    if cross:
        receipt = _verify_cross(folder, pool, np.asarray(bundle.labels), np.asarray(bundle.class_ids),
            np.asarray(list(pool)), np.asarray(pool.gallery), ks, provider, 2026, audit)
    else:
        seen = []
        def replay(query):
            if query != int(provider.query_ids[len(seen)]):
                raise ValueError('Ranking query order differs')
            seen.append(query)
            ids = np.asarray(pool[query])
            return ids, provider.scores(query, ids)
        receipt = ordinary_verify.verify_evaluation(folder, pool, bundle, ks, signature, replay, 2026)
        receipt['metric_query_counts'] = {family: read_json(folder/name)['metric_query_counts']
            for family,name in (('exact','metrics.json'),('coverage','coverage_metrics.json'))}
    filenames = ['run_identity.json','evaluation_manifest.json','rankings.csv','coverage_metrics.json',
                 'coverage_per_query.csv','coverage_per_class.csv']
    if not cross:
        filenames += ['metrics.json','per_query.csv','per_class.csv']
    receipt.update(status='verified', signature=signature, context_signature=manifest['context_signature'],
        stage=stage, split=pool.split_name, variant=style, spec=manifest['spec'], ks=list(ks),
        provider_source=cache_binding, source_binding_sha256=digest(source_binding),
        exact_reports=not cross, candidate_audit=audit,
        ranking_sha256=file_hash(folder/'rankings.csv'), files={name:file_hash(folder/name) for name in filenames},
        scope='All eligible float64 cached scores independently reranked; every query/class/macro/group metric and full-gallery denominator independently recomputed. Regional formulas are separately sampled, not recomputed for every map pair.')
    return receipt


def _evaluate_one(output, stage, style, pool, bundle, provider, context, source_binding,
                  model_name, *, formal=True, ks=KS):
    folder = Path(output)/stage/style
    spec = specifications()[style]
    signature = digest({'context': context, 'stage': stage, 'variant': style, 'spec': spec})
    if not verified(folder, signature):
        _preserve_incomplete(folder, output)
        folder.mkdir(parents=True)
        identity = _identity(output, stage, style, signature)
        cross = stage.startswith('cross_')
        manifest = {**identity, 'signature': signature, 'context_signature': context,
            'model': model_name, 'spec': spec, 'ks': list(ks), 'tie_seed': 2026,
            'num_queries': len(pool), 'gallery_count': len(pool.gallery),
            'candidate_rule': RULE if cross else 'all_gallery_except_query',
            'retrieval_protocol_signature': pool.protocol_signature,
            'base_protocol_signature': provider.protocol_signature,
            'provider_source': {'signature': provider.signature, 'completed_sha256': file_hash(provider.marker)},
            'source_binding_sha256': digest(source_binding), 'ranking_scope': 'top_max_k',
            'training_required': model_name != 'baseline', 'posthoc_research': True,
            'test_parameter_search': False, 'formal_protocol': formal,
            'selection_rule': 'prespecified epoch100; both fixed scoring styles retained',
            'single_seed': 2026}
        if cross:
            manifest['derived_protocol'] = pool.metadata
        record(folder/'run_identity.json', identity)
        record(folder/'evaluation_manifest.json', manifest)
        started = time.perf_counter()
        with (folder/'rankings.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            for query in pool:
                ids = np.asarray(pool[query])
                scores = provider.scores(query, ids)
                order = order_scores(ids, scores, int(query), 2026)[:max(ks)]
                writer.writerows((int(query), rank, int(ids[i]), float(scores[i]))
                                 for rank,i in enumerate(order,1))
        ranking_seconds = time.perf_counter()-started
        if cross:
            coverage_metrics.evaluate_rankings_file(bundle,pool,folder/'rankings.csv',ks,folder,
                                                    split_name=pool.split_name,metadata=manifest)
        else:
            report(folder,pool,bundle,ks,manifest)
        checked = verify_evaluation(folder,pool,bundle,provider,signature,source_binding,formal=formal,ks=ks)
        record(folder/'verification.json',checked)
        record(folder/'timing.json',{'ranking_seconds':ranking_seconds,'total_wall_seconds':time.perf_counter()-started})
        seal(folder,signature)
    manifest=read_json(folder/'evaluation_manifest.json')
    result={'path':str(folder.resolve()),'signature':signature,'key':manifest['key'],'run_id':manifest['run_id'],
            'model':model_name,'style':style,'stage':stage,'split':pool.split_name,
            'register':True,'historical_reference':False,'completed_sha256':file_hash(folder/'completed.json'),
            'coverage_metrics':read_json(folder/'coverage_metrics.json')['metrics'],
            'metric_query_counts':read_json(folder/'verification.json')['metric_query_counts']}
    if not stage.startswith('cross_'):
        result['metrics']=read_json(folder/'metrics.json')['metrics']
    return result


def replay_baseline(output, stage, provider, pool, reference, *, ks=KS):
    """Retain the old identity after exact full-score TopK replay; no new run key."""
    source=Path(reference['folder'])
    if (file_hash(source/'rankings.csv') != reference['ranking_sha256']
            or file_hash(source/'completed.json') != reference['completed_sha256']):
        raise ValueError('Bound historical baseline source changed')
    seen=[]
    with (source/'rankings.csv').open(newline='') as stream:
        for query, group in groupby(csv.DictReader(stream),key=lambda r:int(r['query_id'])):
            rows=list(group)
            ids=np.asarray(pool[query]); scores=provider.scores(query,ids)
            selected=independent_order(query,ids,scores,2026)[:max(ks)]
            if (len(rows)!=max(ks) or [int(r['rank']) for r in rows]!=list(range(1,max(ks)+1))
                    or not np.array_equal(ids[selected],[int(r['candidate_id']) for r in rows])
                    or not np.array_equal(scores[selected],[float(r['similarity_score']) for r in rows])):
                raise ValueError('Historical fusion baseline did not replay exactly: '+stage+'/'+str(query))
            seen.append(query)
    if seen!=list(map(int,provider.query_ids)):
        raise ValueError('Historical baseline query order differs')
    receipt={'status':'verified','stage':stage,'key':reference['key'],'historical_reference':True,
        'register':False,'source':str(source.resolve()),'source_completed_sha256':reference['completed_sha256'],
        'ranking_sha256':reference['ranking_sha256'],'score_cache_sha256':file_hash(provider.marker),
        'num_queries':len(seen),'ranking_rows':len(seen)*max(ks),'topk_identical':True,'max_score_error':0.}
    record(Path(output)/('baseline_replay_'+stage+'.json'),receipt)
    result={**receipt,'path':str(source.resolve()),'model':'baseline','style':'fusion',
        'split':pool.split_name,'signature':read_json(source/'evaluation_manifest.json')['signature'],
        'coverage_metrics':read_json(source/'coverage_metrics.json')['metrics']}
    result['metric_query_counts']={'coverage':read_json(source/'coverage_metrics.json')['metric_query_counts']}
    if not stage.startswith('cross_'):
        result['metrics']=read_json(source/'metrics.json')['metrics']
        result['metric_query_counts']['exact']=read_json(source/'metrics.json')['metric_query_counts']
    return result


def evaluate_model(output, data, model_name, h512_path, context_signature, workers=4,
                   *, formal=True, ks=KS):
    """Evaluate one model on the split loaded by paper_c_ablation.inputs.load.

    ``h512_path`` must contain normalized float32/float64 rows in the unchanged source
    region-token order. Model context and the file hash bind feature provenance;
    feature extraction must separately verify identity with ``token_layout``.
    Returns an immutable index of new evaluations and historical replay references.
    """
    if model_name not in MODELS or not isinstance(context_signature,str) or not context_signature:
        raise ValueError('Unknown model or missing immutable experiment context')
    if type(formal) is not bool or (formal and list(ks)!=KS):
        raise ValueError('Formal evaluations require all seven frozen cutoffs')
    source=data['cache']
    matches=[split for split in ('valid','test') if set(map(int,getattr(data['protocol'].retrieval,split).gallery))
             ==set(map(int,source['map_ids']))]
    if len(matches)!=1:
        raise ValueError('Source region cache does not identify one frozen retrieval split')
    split=matches[0]
    pool=getattr(data['protocol'].retrieval,split)
    cross=make_cross_pool(pool,data['bundle'].labels)
    for candidate_pool in (pool,cross):
        audit_pool(candidate_pool,data['bundle'].labels,ks,formal=formal)
    cache=replaced_cache(source,h512_path)
    if model_name=='baseline' and (cache['features'].dtype!=source['features'].dtype
                                   or not np.array_equal(cache['features'],source['features'])):
        raise ValueError('Baseline must retain every frozen h512 byte; it is not retrained or re-extracted')
    output=Path(output)/'evaluation'/model_name
    science={'parent_context':context_signature,'model':model_name,'split':split,'code':code_identity(),
             'h512_sha256':file_hash(h512_path),'token_layout':token_layout(source),
             'source_binding_sha256':digest(data['source_binding']),'specifications':specifications()}
    context=digest(science)
    record(output/('model_context_'+split+'.json'),{**science,'signature':context})
    formula=audit_formulas(Provider(cache,pool,data['config']['matching']),data['bundle'])
    if formal and (formula['num_classes']!=37 or any(v['candidate_pairs']!=740 for v in formula['variants'].values())):
        raise ValueError('Formal formula audit must cover 37 classes and 740 pairs per style')
    record(output/('formula_audit_'+split+'.json'),formula)
    providers=compute_scores(output,cache,pool,data['config']['matching'],h512_path,context,workers)
    results={}
    for stage,candidate_pool in ((split,pool),('cross_'+split,cross)):
        reference=data.get('replay_references',{}).get(stage,{}).get('primary')
        for style in STYLES:
            if model_name=='baseline' and style=='fusion' and reference:
                result=replay_baseline(output,stage,providers[style],candidate_pool,reference,ks=ks)
            else:
                result=_evaluate_one(output,stage,style,candidate_pool,data['bundle'],providers[style],
                    context,data['source_binding'],model_name,formal=formal,ks=ks)
            results[stage+'/'+style]=result
    if code_identity()!=science['code']:
        raise ValueError('Scoring/report/independent-verification code changed during evaluation')
    index={'signature':context,'context_signature':context_signature,'model':model_name,'split':split,
        'evaluations':results,'score_cache':{style:{'path':str(provider.marker.parent.resolve()),
            'signature':provider.signature,'completed_sha256':file_hash(provider.marker)} for style,provider in providers.items()},
        'formula_audit':str((output/('formula_audit_'+split+'.json')).resolve()),
        'new_evaluations':sum(r['register'] for r in results.values())}
    record(output/('evaluation_index_'+split+'.json'),index)
    return index
