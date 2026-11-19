"""Synthetic immutable WBBS provenance and independent component audit checks."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from fgw_grid_search import sources
from fgw_grid_search.tests.test_cache import Pool, Scorer, W


def signed(value): return {**value, 'signature': digest(value)}


def fixture(root, *, queries=(8,20), gallery=(20,8,99,31)):
    pool = Pool(queries, gallery)
    chosen = sorted(map(int, queries))
    manifest = signed({'context': {'frozen': 'context'}})
    inp = signed({'source': 'frozen'})
    write_json(root / 'run_manifest.json', manifest)
    write_json(root / 'inputs_valid.json', inp)
    cache = root / 'wbbs_cache/valid'
    cache.mkdir(parents=True)
    sig = digest({'inputs': inp['signature'], 'operation': 'full_candidate_W_cache'})
    symmetric = set(queries) == set(gallery)
    plan = {'signature': sig, 'query_ids': list(queries), 'gallery_ids': list(gallery),
            'symmetric_upper_triangle': symmetric, 'audit_ids': chosen, 'dtype': 'float64',
            'backend': 'exact_ckdtree_native_tie_fallback_v1'}
    write_json(cache / 'plan.json', plan)
    nq, ng = len(queries), len(gallery)
    matrix = np.full((nq,ng), np.nan)
    for i,q in enumerate(queries): matrix[i,pool.gallery != q] = W().scores(q,pool[q])
    np.save(cache / 'scores.npy', matrix)
    native = {str(q): {'status': 'verified', 'pairs_checked': ng-1, 'max_error': 0.,
                      'max_abs_error': 0., 'scalar_equality': 'float64 bit-for-bit'} for q in chosen}
    row_hashes = {}
    for q in queries:
        row = cache / 'rows' / str(q)
        row.mkdir(parents=True)
        write_json(row / 'completed.json', {'query_id': q, 'signature': sig})
        row_hashes[str(q)] = file_hash(row / 'completed.json')
    done = {'status': 'completed', 'signature': sig, 'plan_sha256': file_hash(cache / 'plan.json'),
            'scores_sha256': file_hash(cache / 'scores.npy'), 'query_count': nq, 'gallery_count': ng,
            'candidate_count': ng-1, 'scored_pairs': ng*(ng-1)//2 if symmetric else nq*(ng-1),
            'represented_directed_pairs': nq*(ng-1), 'native_audits': native, 'row_marker_hashes': row_hashes}
    write_json(cache / 'completed.json', done)
    source_audit = signed({'status': 'verified', 'split': 'valid',
        'input_context_signature': digest(manifest['context']),
        'wbbs_cache_completed_sha256': file_hash(cache / 'completed.json'),
        'wbbs_native_full_gallery': native,
        'wbbs_all_query_source_replay': {'status': 'verified', 'num_queries': nq,
            'candidates_per_query': ng-1,'top_k': 500,'max_score_error': 0.,'ranking_sha256': 'old-hash'},
        'wbbs_raw_descriptor_replay': {'status': 'verified','query_ids': chosen},
        'fg_full_gallery': {'status': 'verified','query_count': len(chosen), 'queries': [
            {'query_id': q,'candidate_count': ng-1,'all_three_top500_exact': True} for q in chosen]}})
    write_json(root / 'source_audit_valid.json', source_audit)
    ind = {'status': 'verified','split': 'valid','matrix_shape': [nq,ng],'matrix_dtype': 'float64',
        'matrix_sha256': done['scores_sha256'],'cache_completed_sha256': file_hash(cache / 'completed.json'),
        'source_audit_sha256': file_hash(root / 'source_audit_valid.json'),
        'source_audit_signature': source_audit['signature'],'source_context_signature': digest(manifest['context']),
        'frozen_query_ids_order_match': True,'frozen_gallery_ids_order_match': True,
        'row_markers_verified': nq,'row_npz_verified': nq,'row_to_matrix_bitwise_equal': True,
        'candidate_pairs_verified': nq*(ng-1),'self_cells_nan': nq,'native_audit_queries': len(chosen),
        'native_audit_pairs': len(chosen)*(ng-1),'native_audit_max_error': 0.,
        'old_WBBS_top500_queries_replay_evidence': nq,'old_WBBS_top500_source_scores_max_error': 0.,
        'old_WBBS_ranking_sha256': 'old-hash','raw_descriptor_audit_queries': len(chosen),
        'independent_FG_audit_queries': len(chosen),'three_fusion_top500_all_exact': True}
    (root / 'independent_cache_audit').mkdir()
    write_json(root / 'independent_cache_audit/valid.json', ind)
    evaluation = root / 'valid/FGW'
    evaluation.mkdir(parents=True)
    (evaluation / 'rankings.csv').write_text('query_id,rank,candidate_id,similarity_score\n')
    write_json(evaluation / 'completed.json', {'files': {'rankings.csv': file_hash(evaluation / 'rankings.csv')}})
    bound = ['wbbs_cache/valid/plan.json','wbbs_cache/valid/completed.json','source_audit_valid.json',
             'independent_cache_audit/valid.json','valid/FGW/completed.json']
    root_done = {'status': 'completed', 'signature': manifest['signature'],
                 'files': {p: file_hash(root / p) for p in bound}}
    write_json(root / 'completed.json', root_done)
    return pool, chosen, manifest, inp, root_done


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.args = fixture(self.root)
    def tearDown(self): self.temp.cleanup()
    def load(self): return sources.VerifiedWbbsScores(self.root,'valid',*self.args)
    def bind(self, relative):
        self.args[-1]['files'][relative] = file_hash(self.root / relative)
        write_json(self.root / 'completed.json', self.args[-1])
    def test_complete_matrix_load_is_read_only(self):
        before = {str(p): (p.stat().st_mtime_ns,file_hash(p)) for p in self.root.rglob('*') if p.is_file()}
        provider = self.load()
        c = self.args[0][8][::-1]
        np.testing.assert_array_equal(provider.scores(8,c), W().scores(8,c))
        self.assertFalse(provider.matrix.flags.writeable)
        self.assertFalse(provider.provenance['wbbs_recomputed'])
        after = {str(p): (p.stat().st_mtime_ns,file_hash(p)) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_top500_subset_is_not_accepted_as_full_scores(self):
        provider = self.load()
        with self.assertRaises(ValueError): provider.scores(8,[20,99])
    def test_unbound_matrix_hash_change_rejected(self):
        with (self.root / 'wbbs_cache/valid/scores.npy').open('ab') as f: f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'matrix hash'): self.load()
    def test_root_binding_is_required(self):
        self.args[-1]['files'].pop('wbbs_cache/valid/completed.json')
        with self.assertRaisesRegex(ValueError,'no longer bound'): self.load()
    def test_plan_id_permutation_rejected_even_if_rebound(self):
        path = self.root / 'wbbs_cache/valid/plan.json'
        value = read_json(path); value['gallery_ids'] = value['gallery_ids'][::-1]; write_json(path,value)
        self.bind('wbbs_cache/valid/plan.json')
        with self.assertRaises(ValueError): self.load()
    def test_native_missing_or_not_exact_rejected(self):
        path = self.root / 'wbbs_cache/valid/completed.json'
        value = read_json(path); value['native_audits']['8']['max_error'] = 1e-15; write_json(path,value)
        self.bind('wbbs_cache/valid/completed.json')
        with self.assertRaisesRegex(ValueError,'Native WBBS'): self.load()
    def test_previous_independent_audit_must_cover_every_pair(self):
        path = self.root / 'independent_cache_audit/valid.json'
        value = read_json(path); value['candidate_pairs_verified'] -= 1; write_json(path,value)
        self.bind('independent_cache_audit/valid.json')
        with self.assertRaisesRegex(ValueError,'Independent completed'): self.load()
    def test_signed_audit_content_change_rejected(self):
        path = self.root / 'source_audit_valid.json'
        value = read_json(path); value['wbbs_recomputed'] = True; write_json(path,value)
        self.bind('source_audit_valid.json')
        with self.assertRaisesRegex(ValueError,'manifest signature'): self.load()
    def test_missing_or_changed_row_seal_rejected(self):
        path = self.root / 'wbbs_cache/valid/rows/8/completed.json'
        write_json(path, {'bad': True})
        with self.assertRaisesRegex(ValueError,'row marker'): self.load()
    def test_invalid_source_split_rejected_before_loading(self):
        with self.assertRaises(ValueError): sources.load_inputs(self.root, 'train')
    def test_bound_paths_cannot_escape(self):
        with self.assertRaises(ValueError): sources._bound(self.root, {'../evil': 'hash'}, '../evil')


class ComponentAuditTests(unittest.TestCase):
    def test_37_class_independent_fg_and_legacy_ranking_exact(self):
        from b1_ablation.verify import independent_order
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root, queries=tuple(range(37)), gallery=tuple(range(502)))
            pool = args[0]; scorer = Scorer()
            ranking = root / 'valid/FGW/rankings.csv'
            fields = ['query_id','rank','candidate_id','similarity_score','optimal_f_score','global_score','wbbs_score']
            with ranking.open('w',newline='') as handle:
                writer = csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
                for q in pool:
                    c=pool[q]; a=scorer.components(q,c); w=W().scores(q,c)
                    f,g=a['optimal_f_score'],a['global_score']; score=(.5*f+.5*g)*w
                    for rank,i in enumerate(independent_order(q,c,score)[:500],1):
                        writer.writerow(dict(zip(fields,[q,rank,c[i],score[i],f[i],g[i],w[i]])))
            marker=root/'valid/FGW/completed.json'
            write_json(marker,{'files':{'rankings.csv':file_hash(ranking)}})
            args[-1]['files']['valid/FGW/completed.json']=file_hash(marker)
            write_json(root/'completed.json',args[-1])
            provider=sources.VerifiedWbbsScores(root,'valid',*args)
            labels=np.zeros((502,37),dtype=int); labels[:37]=np.eye(37,dtype=int)
            def independent(q,c,sc):
                values=sc.components(q,c)
                return values['optimal_f_score'],values['global_score']
            with patch('fgw_fusion.sources.independent_fg',side_effect=independent):
                result=sources.audit_components(pool,SimpleNamespace(labels=labels),scorer,provider)
                self.assertEqual(result['status'],'verified');self.assertEqual(result['query_count'],37)
                self.assertFalse(result['wbbs_recomputed'])
                # Even a one-bit saved component error is rejected.
                contents=ranking.read_text().replace(',0.0,',',0.0000000000000001,',1)
                ranking.write_text(contents)
                with self.assertRaises(ValueError):
                    sources.audit_components(pool,SimpleNamespace(labels=labels),scorer,provider)


if __name__ == '__main__': unittest.main()
