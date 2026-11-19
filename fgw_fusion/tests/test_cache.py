import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fgw_fusion.cache import build_cache

class Pool(dict):
    split_name='test'
    def __init__(self):
        self.gallery=np.array([11,23,59,71],dtype=np.int64)
        self.candidates_per_query=3
        super().__init__((int(q),self.gallery[self.gallery!=q]) for q in self.gallery)
class Backend:
    def __init__(self):self.calls=0
    def scores(self,q,c):self.calls+=1;return np.array([1/(1+abs(q-int(x))) for x in c])
    def audit_pairs(self,pairs):return {'status':'verified','pairs_checked':sum(1 for _ in pairs)}
class CacheTests(unittest.TestCase):
    def test_upper_triangle_preserves_directed_scores_and_completion_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool=Pool();b=Backend();p,m=build_cache(Path(tmp)/'w','sig',pool,b,workers=1,audit_ids=[11])
            self.assertEqual(m['scored_pairs'],6);self.assertEqual(m['represented_directed_pairs'],12)
            for q,c in pool.items():np.testing.assert_array_equal(p.scores(q,c),Backend().scores(q,c))
            called=b.calls;again,_=build_cache(Path(tmp)/'w','sig',pool,b,workers=1,audit_ids=[11])
            self.assertEqual(b.calls,called)
            with self.assertRaises(ValueError):again.scores(11,np.array([11]))
    def test_completed_hash_corruption_and_context_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'w';pool=Pool();b=Backend()
            build_cache(path,'sig',pool,b,workers=1)
            with self.assertRaises(ValueError):build_cache(path,'changed',pool,b,workers=1)
            with (path/'scores.npy').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):build_cache(path,'sig',pool,b,workers=1)
    def test_interrupted_rows_are_reused_without_rescoring(self):
        class Stop(Backend):
            def scores(self,q,c):
                if q==59:raise RuntimeError('simulate interruption')
                return super().scores(q,c)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'w';pool=Pool()
            with self.assertRaises(RuntimeError):build_cache(path,'sig',pool,Stop(),workers=1)
            before=(path/'rows/11/scores.npz').read_bytes();b=Backend()
            p,m=build_cache(path,'sig',pool,b,workers=1)
            self.assertEqual(b.calls,2);self.assertEqual(before,(path/'rows/11/scores.npz').read_bytes())
            self.assertEqual(m['query_count'],4)
    def test_valid_subset_uses_full_gallery_and_no_triangular_shortcut(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool=Pool();pool.split_name='valid';del pool[23];del pool[71]
            p,m=build_cache(Path(tmp)/'w','sig',pool,Backend(),workers=1)
            self.assertEqual(m['scored_pairs'],6)
            for q,c in pool.items():np.testing.assert_array_equal(p.scores(q,c),Backend().scores(q,c))


class CacheRecoveryTests(unittest.TestCase):
    def test_resume_after_matrix_publish_before_completion_preserves_orphan_and_rows(self):
        from unittest.mock import patch
        from fgw_fusion import cache
        from wbm_retrieval.common import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"w";pool=Pool();original_write=cache.write_json
            def stop_before_seal(destination,value):
                if Path(destination)==path/"completed.json":
                    self.assertTrue((path/"scores.npy").is_file())
                    raise RuntimeError("interrupted between matrix publish and completion")
                original_write(destination,value)
            with patch.object(cache,"write_json",side_effect=stop_before_seal):
                with self.assertRaisesRegex(RuntimeError,"between matrix publish"):
                    build_cache(path,"sig",pool,Backend(),workers=1,audit_ids=[11])
            self.assertFalse((path/"completed.json").exists())
            orphan_bytes=(path/"scores.npy").read_bytes()
            row_snapshot={str(f.relative_to(path)): (file_hash(f),f.stat().st_mtime_ns)
                          for f in (path/"rows").rglob("*") if f.is_file()}
            class MustNotScore(Backend):
                def scores(self,q,c):raise AssertionError("completed row rescored")
                def audit_pairs(self,pairs):raise AssertionError("completed audit repeated")
            provider,marker=build_cache(path,"sig",pool,MustNotScore(),workers=1,audit_ids=[11])
            self.assertEqual(marker["status"],"completed")
            preserved=list(path.glob(".preserved_unsealed_matrix_*/scores.npy"))
            self.assertEqual(len(preserved),1)
            self.assertEqual(preserved[0].read_bytes(),orphan_bytes)
            self.assertTrue((preserved[0].parent/"reason.json").is_file())
            self.assertEqual(row_snapshot,{str(f.relative_to(path)): (file_hash(f),f.stat().st_mtime_ns)
                                          for f in (path/"rows").rglob("*") if f.is_file()})
            for q,c in pool.items():np.testing.assert_array_equal(provider.scores(q,c),Backend().scores(q,c))

    def test_completed_cache_missing_or_invalid_native_audits_are_rejected(self):
        import copy
        from wbm_retrieval.common import read_json,write_json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"w";pool=Pool();backend=Backend()
            build_cache(path,"sig",pool,backend,workers=1,audit_ids=[11])
            complete=read_json(path/"completed.json");calls=backend.calls
            missing=copy.deepcopy(complete);missing.pop("native_audits")
            empty=copy.deepcopy(complete);empty["native_audits"]={}
            failed=copy.deepcopy(complete);failed["native_audits"]["11"]["status"]="failed"
            partial=copy.deepcopy(complete);partial["native_audits"]["11"]["pairs_checked"]=2
            extra=copy.deepcopy(complete);extra["native_audits"]["23"]={"status":"verified","pairs_checked":3}
            wrong_count=copy.deepcopy(complete);wrong_count["query_count"]=3
            for index,marker in enumerate((missing,empty,failed,partial,extra,wrong_count)):
                with self.subTest(case=index):
                    write_json(path/"completed.json",marker)
                    with self.assertRaisesRegex(ValueError,"audit/identity incomplete"):
                        build_cache(path,"sig",pool,backend,workers=1,audit_ids=[11])
            self.assertEqual(backend.calls,calls)

    def test_completed_matrix_shape_and_dtype_rejected_even_with_matching_hash(self):
        from wbm_retrieval.common import read_json,write_json,file_hash
        for shape,dtype in (((4,3),np.float64),((4,4),np.float32)):
            with self.subTest(shape=shape,dtype=str(dtype)),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"w";pool=Pool();backend=Backend()
                build_cache(path,"sig",pool,backend,workers=1)
                np.save(path/"scores.npy",np.zeros(shape,dtype=dtype),allow_pickle=False)
                marker=read_json(path/"completed.json");marker["scores_sha256"]=file_hash(path/"scores.npy")
                write_json(path/"completed.json",marker)
                with self.assertRaisesRegex(ValueError,"shape/dtype differs"):
                    build_cache(path,"sig",pool,backend,workers=1)

    def test_interrupted_cached_row_requires_full_native_audit(self):
        from wbm_retrieval.common import read_json,write_json
        class Stop(Backend):
            def scores(self,q,c):
                if q==59:raise RuntimeError("stop before all rows")
                return super().scores(q,c)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"w";pool=Pool()
            with self.assertRaises(RuntimeError):
                build_cache(path,"sig",pool,Stop(),workers=1,audit_ids=[11])
            row=path/"rows/11/completed.json";marker=read_json(row);marker["native_audit"]=None
            write_json(row,marker)
            with self.assertRaisesRegex(ValueError,"Native audit missing"):
                build_cache(path,"sig",pool,Backend(),workers=1,audit_ids=[11])

if __name__=='__main__':unittest.main()
