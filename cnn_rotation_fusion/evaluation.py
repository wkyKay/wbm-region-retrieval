"""Twelve regional additive formulas with canonical references and independent audits.

The original h512 token ordering, raw-mask structure/area and greedy matching stay
fixed. Complete float64 score rows feed ordinary and cross-combination rankings.
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
from f_region_wbbs.protocol import KS, PROTOCOL, RULE, FROZEN, BASE_PROTOCOL_SIGNATURE, audit_pool, make_cross_pool
from f_region_wbbs.verify import _verify_cross
from fgw_grid_search.verify import independent_order
from paper_c_ablation.pipeline import CachedProvider, read_scoring_cache
from paper_c_ablation.scoring import normalize_matching
from wbm_retrieval.common import digest, file_hash, read_json
from wbm_retrieval.diagnostics import batched_map_scores
from wbm_retrieval.exact_metrics import order_scores

from .state import MODELS, GRID, THETA0, KS, signed, read_signed, check_binding
from f_region_area_fusion.pipeline import audit_formulas
from cnn_augmentation.evaluation import token_layout, replaced_cache, Provider as OriginalProvider

FIELDS = ('query_id', 'rank', 'candidate_id', 'similarity_score')

def specifications():
    return GRID

class Provider(OriginalProvider):
    @staticmethod
    def pair_scores(parts, variant):
        spec = GRID[variant] if isinstance(variant, str) else variant
        if spec not in GRID.values():
            raise ValueError('Only the frozen twelve additive weights are permitted')
        return historical_fusion(parts, spec)


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


def compute_scores(output, cache, pool, matching, h512_path, model_context, variants, workers=4):
    output, h512_path = Path(output), Path(h512_path).resolve()
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Use 1 to 16 evaluation workers')
    specs = {name: GRID[name] for name in variants}
    source_sha = file_hash(Path(cache['folder'])/'completed.json')
    feature_sha = file_hash(h512_path)
    layout = token_layout(cache)
    paths, folders = {}, {}
    for style in specs:
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
    key = 'cnn_rotation_fusion_'+re.sub(r'[^A-Za-z0-9_]+', '_', str(Path(output).parents[1].name+'_'+Path(output).name))+'_'+stage+'_'+style
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
            'training_required': False, 'posthoc_research': True,
            'test_parameter_search': False, 'formal_protocol': formal,
            'selection_rule': 'twelve ordinary-valid weights; integer exact Top10 hits; unique frozen winner',
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
            'model':model_name,'variant':style,'spec':spec,'stage':stage,'split':pool.split_name,
            'register':True,'historical_reference':False,'completed_sha256':file_hash(folder/'completed.json'),
            'coverage_metrics':str((folder/'coverage_metrics.json').resolve()),
            'source_binding':{str(folder/'completed.json'):file_hash(folder/'completed.json')},
            'metric_query_counts':read_json(folder/'verification.json')['metric_query_counts']}
    if not stage.startswith('cross_'):
        result['metrics']=str((folder/'metrics.json').resolve())
    return result



def evaluate_variants(output, data, model, split, variants, stages, context_signature, workers=4,
                      *, formal=True, ks=KS, replay_theta0=False):
    """Compute needed full score rows once; reuse canonical evaluations unchanged."""
    if model not in MODELS or split not in ('valid','test') or not set(variants) <= set(GRID):
        raise ValueError('Unknown model/split/weight')
    if not set(stages) <= {split,'cross_'+split} or len(set(variants)) != len(variants):
        raise ValueError('Invalid stage or duplicate weight')
    if formal and list(ks) != KS:
        raise ValueError('Formal evaluation requires all seven cutoffs')
    output = Path(output)
    folder = output/'evaluation'/model
    pool = getattr(data['protocol'].retrieval,split)
    cross = make_cross_pool(pool,data['bundle'].labels)
    for p in (pool,cross): audit_pool(p,data['bundle'].labels,ks,formal=formal)
    feature = Path(data['features'][model])
    science = {'parent_context':context_signature,'model':model,'split':split,
        'h512_path':str(feature.resolve()), 'h512_sha256':file_hash(feature),
        'token_layout':token_layout(data['cache']), 'source_binding_sha256':digest(data['source_binding']),
        'specifications':GRID}
    context = digest(science)
    record(folder/f'model_context_{split}.json',{**science,'signature':context})
    needed = sorted({name for stage in stages for name in variants
        if model+'/'+stage+'/'+name not in data['references']} | ({THETA0} if replay_theta0 else set()))
    providers = {}
    if needed:
        cache = replaced_cache(data['cache'],feature)
        formula = audit_formulas(Provider(cache,pool,data['config']['matching']),data['bundle'],folder,GRID,context)
        if formal and formula['candidate_pairs'] != 37*20*12:
            raise ValueError('Formula audit must cover all classes and twelve variants')
        providers = compute_scores(folder,cache,pool,data['config']['matching'],feature,context,needed,workers)
    results = {}
    for stage in stages:
        for name in variants:
            role = model+'/'+stage+'/'+name
            if role in data['references']:
                results[role] = data['references'][role]
            else:
                row = _evaluate_one(folder,stage,name,cross if stage.startswith('cross_') else pool,
                    data['bundle'],providers[name],context,data['source_binding'],model,formal=formal,ks=ks)
                row['provider_path'] = str(providers[name].marker.parent.resolve())
                row['model_context_path'] = str((folder/f'model_context_{split}.json').resolve())
                results[role] = row
    if replay_theta0:
        if split != 'valid' or model != 'rotation':
            raise ValueError('Compatibility replay is the rotation ordinary-valid theta0 anchor')
        row = data['references'][model+'/'+split+'/'+THETA0]
        provider = providers[THETA0]
        receipt_path = output/'theta0_replay.json'
        fixed = {'context_signature':context_signature,'model_context':context,'reference_key':row['key'],
                 'reference_completed_sha256':row['completed_sha256'], 'provider_signature':provider.signature}
        if receipt_path.exists():
            receipt = read_signed(receipt_path)
            if any(receipt.get(k) != v for k,v in fixed.items()):
                raise ValueError('Theta0 compatibility replay changed')
        else:
            def replay(query):
                ids = np.asarray(pool[query]); return ids,provider.scores(query,ids)
            checked = ordinary_verify.verify_evaluation(Path(row['path']),pool,data['bundle'],ks,row['signature'],replay,2026)
            if checked['max_saved_score_absolute_error'] != 0:
                raise ValueError('Theta0 replay requires byte-identical saved scores and rankings')
            record(receipt_path,signed({**fixed,'verification':checked,
                'status':'verified','scope':'all valid queries/full candidates; same Top500 IDs and exact saved scores; no new evaluation identity'}))
    check_binding(data['source_binding'])
    return results


def verify_evaluation_row(row, *, check_sources=True):
    """Read-only seal/identity verification; full reranking was sealed at creation."""
    folder = Path(row['path'])
    if row.get('register') is row.get('historical_reference') or row.get('spec') != GRID.get(row.get('variant')):
        raise ValueError('Invalid evaluation role or parameter identity')
    if not verified(folder,row['signature']) or file_hash(folder/'completed.json') != row['completed_sha256']:
        raise ValueError('Evaluation seal changed')
    meta = read_json(folder/'evaluation_manifest.json')
    identity = read_json(folder/'run_identity.json')
    check = read_json(folder/'verification.json')
    if (row['key'] != identity['key'] or row['signature'] != identity['evaluation_signature']
            or identity['run_id'] != row.get('run_id') or meta['signature'] != row['signature']
            or check.get('status') != 'verified' or check.get('all_queries_full_candidate_topk_exact') is not True):
        raise ValueError('Report identity or independent full-candidate proof changed')
    cross = row['stage'].startswith('cross_')
    if row['split'] != row['stage'].removeprefix('cross_') or meta['split'] != row['split']:
        raise ValueError('Report stage/split mismatch')
    if row['register']:
        context = read_signed(row['model_context_path'])
        provider = CachedProvider(row['provider_path'])
        cache_meta = read_json(provider.marker.parent/'metadata.json')
        if (cache_meta['context_signature'] != context['signature'] or cache_meta['variant'] != row['variant']
                or cache_meta['spec'] != row['spec'] or cache_meta['split'] != row['split']
                or cache_meta['h512_sha256'] != context['h512_sha256'] or cache_meta['token_layout'] != context['token_layout']):
            raise ValueError('Score cache detached from feature/geometry/parameter context')
        expected = digest({'context':context['signature'],'stage':row['stage'],'variant':row['variant'],'spec':row['spec']})
        if (meta.get('model') != row['model'] or meta.get('variant') != row['variant'] or meta.get('spec') != row['spec']
                or meta.get('training_required') is not False or meta.get('test_parameter_search') is not False
                or meta['signature'] != expected or meta['context_signature'] != context['signature']
                or meta['source_binding_sha256'] != context['source_binding_sha256']
                or meta['provider_source'] != {'signature':provider.signature,'completed_sha256':file_hash(provider.marker)}):
            raise ValueError('New evaluation detached from model/weight/score provenance')
    if meta.get('formal_protocol',True):
        frozen = FROZEN[row['split']]
        nq, ng = frozen['num_queries'], frozen['gallery_count']
        if (meta.get('ks') != KS or meta.get('tie_seed') != 2026 or meta.get('num_queries') != nq
                or meta.get('gallery_count') != ng
                or meta.get('retrieval_protocol_signature') != (frozen['derived_signature'] if cross else BASE_PROTOCOL_SIGNATURE)
                or meta.get('candidate_rule') != (RULE if cross else 'all_gallery_except_query')
                or check.get('num_queries') != nq or check.get('ranking_rows') != nq*max(KS)
                or check.get('full_candidates_checked') != (frozen['candidate_pairs'] if cross else nq*(ng-1))):
            raise ValueError('Frozen full-gallery protocol/counts differ')
    if check.get('ranking_sha256') != file_hash(folder/'rankings.csv'):
        raise ValueError('Independent verification no longer binds rankings')
    if check_sources: check_binding(row.get('source_binding',{}))
    families = [('coverage','coverage_metrics','coverage_metrics.json')]
    if not cross: families.append(('exact','metrics','metrics.json'))
    elif 'metrics' in row or (folder/'metrics.json').exists():
        raise ValueError('Cross role cannot contain exact metrics')
    for family,key,name in families:
        if Path(row[key]).resolve() != (folder/name).resolve():
            raise ValueError('Metric path detached from evaluated folder')
        report_data = read_json(folder/name)
        if (any(report_data['metadata'].get(k)!=v for k,v in meta.items())
                or row['metric_query_counts'][family] != report_data['metric_query_counts']):
            raise ValueError('Report metadata/counts differ from canonical evaluation')
    return row


def verify_index(path, *, context_signature=None, check_sources=True):
    index = read_signed(path)
    if context_signature is not None and index['context_signature'] != context_signature:
        raise ValueError('Index detached from root')
    for role,row in index['evaluations'].items():
        if role != row['model']+'/'+row['stage']+'/'+row['variant']:
            raise ValueError('Evaluation index role mismatch')
        verify_evaluation_row(row,check_sources=check_sources)
    return index
