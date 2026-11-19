"""Full-gallery inclusion, paired metrics, SHA ordering and immutable recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from fgw_fusion import pipeline, scoring, verify
from b1_structure_fusion.verify import independent_query_metrics
from b1_ablation.verify import independent_order
from experiments.proposed.f_segmentation_geometry import pipeline as geo
from wbm_retrieval.common import file_hash, read_json
from wbm_retrieval.exact_metrics import order_scores
from wbm_retrieval.retrieval_protocol import FullGalleryPool


class Components:
    def __init__(self, fail=None):self.calls=[];self.fail=fail
    def components(self, query, candidates):
        if query == self.fail:raise InterruptedError("query interruption")
        self.calls.append((int(query),tuple(map(int,candidates))))
        values=np.where(np.asarray(candidates)==11,1.,.1)
        return {"optimal_f_score":values,"global_score":values,
                "angular_distance":np.ones(len(values))}


class WBBS:
    def __init__(self):self.calls=[]
    def scores(self, query, candidates):
        self.calls.append((int(query),tuple(map(int,candidates))))
        return np.where(np.asarray(candidates)==11,.5,1.)


def fixture(split="valid"):
    ids=np.arange(12,dtype=np.int64)
    labels=np.tile([0,1,0,0,0,0,0,0],(12,1));labels[[0,11]]=[1,0,0,0,0,0,0,0]
    labels[2]=[1,1,0,0,0,0,0,0]
    _,classes=np.unique(labels,axis=0,return_inverse=True)
    bundle=SimpleNamespace(labels=labels,class_ids=classes,valid_ids=ids,reserved_ids=ids)
    pool=FullGalleryPool([0,1,2],ids,split,"synthetic_frozen_protocol")
    settings=deepcopy(scoring.SETTINGS);settings["ks"]=[1,5,10]
    return pool,bundle,settings


def snapshot(folder):
    return {str(p.relative_to(folder)):(file_hash(p),p.stat().st_mtime_ns)
            for p in Path(folder).rglob("*") if p.is_file()}


def evaluate(out,pool,bundle,settings,components=None,wbbs=None,signature="source"):
    return pipeline.evaluate(out,signature,components or Components(),pool,bundle,
                             settings["ks"],2026,wbbs or WBBS(),settings)


class PipelineTests(unittest.TestCase):
    def test_all_candidates_can_enter_new_fusion_and_exact_coverage_share_ranking(self):
        pool,bundle,settings=fixture();scorer=Components();wbbs=WBBS()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            out=Path(tmp)/"fgw"
            reports=evaluate(out,pool,bundle,settings,scorer,wbbs)
            self.assertEqual(set(reports),{"FGW","GW","FW"})
            self.assertEqual(len(scorer.calls),len(pool));self.assertEqual(scorer.calls,wbbs.calls)
            self.assertEqual(scorer.calls[0][1],tuple(range(1,12)))
            candidates=pool[0]
            w=WBBS().scores(0,candidates)
            old_top=candidates[order_scores(candidates,w,0,2026)[:10]]
            self.assertNotIn(11,old_top)
            for name in scoring.specs():
                folder=out/"valid"/name
                with (folder/"rankings.csv").open() as f:rows=[r for r in csv.DictReader(f) if r["query_id"]=="0"]
                self.assertEqual(int(rows[0]["candidate_id"]),11)
                ranked=np.array([int(r["candidate_id"]) for r in rows])
                actual=independent_query_metrics(bundle.labels[0],bundle.labels[candidates],bundle.labels[ranked],settings["ks"])
                for family,filename in (("exact","per_query.csv"),("coverage","coverage_per_query.csv")):
                    with (folder/filename).open() as f:row=next(r for r in csv.DictReader(f) if r["query_id"]=="0")
                    for key,value in actual[family].items():
                        if value is not None:self.assertAlmostEqual(float(row[key]),value,places=14)
            index=read_json(out/"evaluation_index_valid.json")
            identities=[]
            for name in scoring.specs():
                folder=out/"valid"/name
                self.assertTrue(geo.completed(folder,index["variant_signatures"][name]))
                checked=verify.verify_evaluation(folder,pool,bundle,settings["ks"],index["variant_signatures"][name])
                self.assertEqual(checked["status"],"verified");self.assertEqual(checked["num_queries"],len(pool))
                self.assertEqual(file_hash(folder/"completed.json"),index["completed_sha256"][name])
                identity=read_json(folder/"run_identity.json")
                self.assertTrue(identity["key"].startswith("fgw_"));self.assertTrue(identity["started_at"].endswith("+08:00"))
                identities.append(identity["run_id"])
            self.assertEqual(len(set(identities)),3)

    def test_exact_sha_ties_without_candidate_order_leakage(self):
        pool,bundle,settings=fixture()
        class EqualComponents:
            def components(self,q,c):return {"optimal_f_score":np.ones(len(c)),"global_score":np.ones(len(c))}
        class EqualWBBS:
            def scores(self,q,c):return np.ones(len(c))
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            out=Path(tmp)/"fgw";evaluate(out,pool,bundle,settings,EqualComponents(),EqualWBBS())
            with (out/"valid/FGW/rankings.csv").open() as f:rows=[r for r in csv.DictReader(f) if r["query_id"]=="0"]
            expected=pool[0][independent_order(0,pool[0],np.ones(len(pool[0])))[:10]]
            np.testing.assert_array_equal([int(r["candidate_id"]) for r in rows],expected)

    def test_interrupted_query_reuses_only_verified_shard_and_original_identity(self):
        pool,bundle,settings=fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            out=Path(tmp)/"fgw"
            with self.assertRaises(InterruptedError):evaluate(out,pool,bundle,settings,Components(fail=1))
            shard_before=snapshot(out/"shards/valid/0");identities=snapshot(out/"identities")
            attempts={str(p):snapshot(p) for p in (out/"valid").glob(".*attempt_*")}
            resumed=Components();w=WBBS();evaluate(out,pool,bundle,settings,resumed,w)
            self.assertEqual([q for q,_ in resumed.calls],[1,2]);self.assertEqual(resumed.calls,w.calls)
            self.assertEqual(shard_before,snapshot(out/"shards/valid/0"));self.assertEqual(identities,snapshot(out/"identities"))
            self.assertEqual(attempts,{p:snapshot(Path(p)) for p in attempts})

    def test_metrics_interruption_preserves_attempts_and_reuses_all_query_shards(self):
        pool,bundle,settings=fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            out=Path(tmp)/"fgw"
            with patch.object(geo,"evaluate_rankings_file",side_effect=InterruptedError("after exact")):
                with self.assertRaises(InterruptedError):evaluate(out,pool,bundle,settings)
            attempts={str(p):snapshot(p) for p in (out/"valid").glob(".*attempt_*")}
            identities=snapshot(out/"identities");shards=snapshot(out/"shards")
            with patch.object(Components,"components",side_effect=AssertionError("must reuse full candidate shard")):
                evaluate(out,pool,bundle,settings)
            self.assertEqual(identities,snapshot(out/"identities"));self.assertEqual(shards,snapshot(out/"shards"))
            self.assertEqual(attempts,{p:snapshot(Path(p)) for p in attempts})
            before={name:snapshot(out/"valid"/name) for name in scoring.specs()}
            with patch.object(Components,"components",side_effect=AssertionError("must reuse completed")):
                evaluate(out,pool,bundle,settings)
            self.assertEqual(before,{name:snapshot(out/"valid"/name) for name in scoring.specs()})
            with self.assertRaisesRegex(ValueError,"signature changed"):
                evaluate(out,pool,bundle,settings,signature="different_source")
            (out/"valid/FGW/rankings.csv").write_text("corrupt")
            with self.assertRaises(ValueError):evaluate(out,pool,bundle,settings)

    def test_corrupt_or_different_source_query_shard_is_refused(self):
        pool,bundle,settings=fixture()
        for corrupt in (False,True):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
                out=Path(tmp)/"fgw"
                with self.assertRaises(InterruptedError):evaluate(out,pool,bundle,settings,Components(fail=1))
                if corrupt:
                    (out/"shards/valid/0/scores.npz").write_bytes(b'corrupt')
                    with self.assertRaisesRegex(ValueError,"hash mismatch"):evaluate(out,pool,bundle,settings)
                else:
                    with self.assertRaisesRegex(ValueError,"signature changed"):
                        evaluate(out,pool,bundle,settings,signature="changed")

    def test_valid_test_isolation(self):
        pool,bundle,settings=fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(StringIO()):
            out=Path(tmp)/"fgw";evaluate(out,pool,bundle,settings)
            before=snapshot(out/"valid")
            p,b,s=fixture("test");evaluate(out,p,b,s,signature="test_source")
            self.assertEqual(before,snapshot(out/"valid"))
            with (out/"comparison.csv").open() as f:rows=list(csv.DictReader(f))
            self.assertEqual({r["split"] for r in rows},{"valid","test"})
            self.assertEqual({r["variant"] for r in rows},{"FGW","GW","FW"})

    def test_rejects_wrong_pool_settings_duplicates_self_and_missing_wbbs_scores(self):
        pool,bundle,settings=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"K/tie"):
                pipeline.evaluate(tmp,"sig",Components(),pool,bundle,[1],2026,WBBS(),settings)
            with self.assertRaisesRegex(ValueError,"K/tie"):
                pipeline.evaluate(tmp,"sig",Components(),pool,bundle,settings["ks"],42,WBBS(),settings)
        class UnsafePool:
            split_name="valid";gallery=pool.gallery;candidates_per_query=11;protocol_signature="unsafe"
            def __len__(self):return 1
            def __iter__(self):return iter([0])
            def __getitem__(self,q):return np.arange(11)
        class DuplicatePool(UnsafePool):
            def __getitem__(self,q):return np.array(list(range(1,11))+[10])
        for bad in (UnsafePool(),DuplicatePool()):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError,"entire gallery except self"):
                    evaluate(tmp,bad,bundle,settings)
        class WrongWBBS:
            def scores(self,q,c):return np.ones(len(c)-1)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"unit-interval"):
                evaluate(tmp,pool,bundle,settings,wbbs=WrongWBBS())


if __name__ == "__main__":unittest.main()
