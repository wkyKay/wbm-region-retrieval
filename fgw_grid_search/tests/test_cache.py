"""Synthetic complete-pool, atomic resume and corruption checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wbm_retrieval.common import file_hash, read_json, write_json
from fgw_grid_search.cache import FullScoreCache, build_cache


class Pool:
    split_name = 'valid'
    protocol_signature = 'frozen-synthetic'
    def __init__(self, queries=(8, 20), gallery=(20, 8, 99, 31)):
        self.queries = np.array(queries, dtype=np.int64)
        self.gallery = np.array(gallery, dtype=np.int64)
        self.candidates_per_query = len(gallery) - 1
    def __iter__(self): return iter(self.queries)
    def __len__(self): return len(self.queries)
    def __getitem__(self, q): return self.gallery[self.gallery != q]


class Scorer:
    def __init__(self): self.calls = []
    def components(self, q, c):
        self.calls.append(q)
        return {'optimal_f_score': (c % 17).astype(float) / 17,
                'global_score': np.full(c.shape, q / 100, dtype=float)}


class W:
    def __init__(self, fail=None): self.calls, self.fail = [], fail
    def scores(self, q, c):
        self.calls.append(q)
        if q == self.fail: raise RuntimeError('synthetic interruption')
        return (c % 13).astype(float) / 13


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / 'cache'
        self.pool, self.scorer, self.w = Pool(), Scorer(), W()
    def tearDown(self): self.temp.cleanup()
    def build(self, **kwargs):
        return build_cache(self.folder, 'sig', self.pool, self.scorer, self.w, **kwargs)
    def test_full_original_identity_order_and_float64(self):
        seen = []
        provider, marker = self.build(audit_callback=lambda q,c,v: seen.append(q) or {'verified': True})
        self.assertEqual(seen, [8, 20])
        self.assertEqual(provider.marker, self.folder / 'completed/completed.json')
        self.assertEqual(provider.signature, 'sig')
        self.assertEqual(provider.protocol_signature, self.pool.protocol_signature)
        self.assertEqual(marker['pairs_per_component'], 6)
        c = self.pool[8][::-1]
        scores = provider.scores(8, c)
        np.testing.assert_array_equal(scores['F'], (c % 17) / 17.)
        np.testing.assert_array_equal(scores['WBBS'], (c % 13) / 13.)
        for a in provider.matrices.values():
            self.assertEqual(a.dtype, np.float64)
            self.assertFalse(a.flags.writeable)
            self.assertTrue(np.isnan(a[provider.rows[8], provider.columns[8]]))
        for q in self.pool:
            self.assertTrue(read_json(self.folder / 'rows' / str(q) / 'completed.json')['audit']['verified'])
    def test_complete_reopen_is_read_only_and_never_recomputes(self):
        self.build()
        before = {str(p): (p.stat().st_mtime_ns, file_hash(p)) for p in self.folder.rglob('*') if p.is_file()}
        self.scorer.calls.clear(); self.w.calls.clear()
        self.build(audit_callback=lambda *a: self.fail('Should not rerun audit'))
        FullScoreCache(self.folder)
        after = {str(p): (p.stat().st_mtime_ns, file_hash(p)) for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.scorer.calls, [])
        self.assertEqual(self.w.calls, [])
    def test_incomplete_resume_reuses_only_verified_sealed_rows(self):
        self.w.fail = 20
        with self.assertRaises(RuntimeError): self.build()
        oldrow = self.folder / 'rows/8/row.npz'
        before = (oldrow.stat().st_mtime_ns, file_hash(oldrow))
        orphan = self.folder / '.attempt_matrices_preserved'
        orphan.mkdir(); (orphan / 'partial').write_text('unsealed')
        self.w.fail = None; self.scorer.calls.clear(); self.w.calls.clear()
        self.build()
        self.assertEqual(self.scorer.calls, [20])
        self.assertEqual(self.w.calls, [20])
        self.assertEqual(before, (oldrow.stat().st_mtime_ns, file_hash(oldrow)))
        self.assertEqual((orphan / 'partial').read_text(), 'unsealed')
    def test_corrupt_row_on_resume_is_rejected_without_recompute(self):
        self.w.fail = 20
        with self.assertRaises(RuntimeError): self.build()
        row = self.folder / 'rows/8/row.npz'
        with row.open('ab') as f: f.write(b'bad')
        self.scorer.calls.clear(); self.w.fail = None
        with self.assertRaisesRegex(ValueError, 'row identity/hash'): self.build()
        self.assertEqual(self.scorer.calls, [])
    def test_unsealed_named_row_is_preserved_and_rejected(self):
        self.w.fail = 20
        with self.assertRaises(RuntimeError): self.build()
        row = self.folder / 'rows/20'
        row.mkdir(); (row / 'partial').write_text('keep')
        with self.assertRaises(FileNotFoundError): self.build()
        self.assertEqual((row / 'partial').read_text(), 'keep')
    def test_corrupt_final_matrix_is_rejected(self):
        self.build()
        with (self.folder / 'completed/F.npy').open('ab') as f: f.write(b'bad')
        with self.assertRaisesRegex(ValueError, 'file hash'): FullScoreCache(self.folder)
    def test_rehashed_invalid_matrix_or_id_file_is_rejected(self):
        for kind in ('self', 'nan', 'range', 'dtype', 'ids'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / 'cache'
                build_cache(folder, 'sig', self.pool, Scorer(), W())
                filename = 'query_ids.npy' if kind == 'ids' else 'F.npy'
                p = folder / 'completed' / filename
                a = np.load(p)
                if kind == 'self': a[0, 1] = .2
                elif kind == 'nan': a[0, 0] = np.nan
                elif kind == 'range': a[0, 0] = 1.1
                elif kind == 'dtype': a = a.astype(np.float32)
                else: a = a[::-1]
                np.save(p, a)
                marker = read_json(folder / 'completed/completed.json')
                marker['files'][filename] = file_hash(p)
                write_json(folder / 'completed/completed.json', marker)
                with self.assertRaises(ValueError): FullScoreCache(folder)
    def test_signature_or_protocol_change_rejected(self):
        self.build()
        with self.assertRaises(ValueError): build_cache(self.folder, 'new', self.pool, self.scorer, self.w)
        self.pool.protocol_signature = 'changed'
        with self.assertRaises(ValueError): self.build()
    def test_all_candidate_violations_rejected(self):
        provider, _ = self.build()
        for q, c in [(8,[20,99]),(8,[20,99,99]),(8,[8,20,99]),(8,[20,99,100]),
                     (8,[20.,99.,31.]),(99,[20,8,31]),(8.5,[20,99,31]),(True,[20,99,31])]:
            with self.subTest(q=q,c=c), self.assertRaises(ValueError): provider.scores(q, c)
    def test_invalid_fresh_scores_do_not_seal_a_row(self):
        self.w.scores = lambda q,c: np.full(c.shape, np.nan)
        with self.assertRaises(ValueError): self.build()
        self.assertFalse((self.folder / 'rows/8').exists())
        self.assertFalse((self.folder / 'completed').exists())
    def test_unknown_unsealed_cache_is_never_adopted(self):
        self.folder.mkdir(); (self.folder / 'untouched').write_text('old')
        with self.assertRaises(ValueError): self.build()
        self.assertEqual((self.folder / 'untouched').read_text(), 'old')
    def test_test_pool_uses_full_directed_rows(self):
        self.pool = Pool((20,8,99,31)); self.pool.split_name = 'test'
        provider, marker = self.build()
        self.assertEqual(marker['pairs_per_component'], 12)
        self.assertEqual(len(self.w.calls), 4)
        np.testing.assert_array_equal(provider.query_ids, self.pool.gallery)


if __name__ == '__main__': unittest.main()
