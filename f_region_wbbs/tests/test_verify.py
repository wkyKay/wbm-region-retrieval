"""Four-stage real evaluator fixtures and independent regional BBP audits."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f_region_evidence.tests.test_verify import fixture as evidence_fixture
from f_region_evidence.tests.test_scoring import fixture as region_fixture
from fgw_cross_test.tests.test_verify import rewrite_csv, write_json
from f_region_wbbs import scoring, state, verify


def fixture(root,stage="cross_valid",name="region_j"):
    cross=stage.startswith("cross_")
    values=list(evidence_fixture(root,cross=cross))
    folder,pool,bundle,ks,_,provider,binding=values
    split="valid" if stage.endswith("valid") else "test"
    pool.split_name=split
    bundle.valid_ids=bundle.reserved_ids=pool.gallery
    spec={"family":"global_wbbs"} if name=="global_wbbs" else state.grid()[name]
    context="synthetic-regional-wbbs-context"
    signature=state.evaluation_signature(context,stage,name,spec)
    manifest=verify._json(folder/"evaluation_manifest.json")
    identity=verify._json(folder/"run_identity.json")
    identity.update(key="f_region_wbbs_synthetic_"+stage,variant=name,stage=stage,split=split,
                    evaluation_signature=signature)
    identity["run_id"]="20261120T210203+0800__"+identity["key"]
    manifest.update(identity)
    manifest.update(signature=signature,context_signature=context,spec=spec)
    write_json(folder/"evaluation_manifest.json",manifest);write_json(folder/"run_identity.json",identity)
    for filename in (["coverage_metrics.json"] if cross else ["metrics.json","coverage_metrics.json"]):
        path=folder/filename;report=verify._json(path)
        report.update(split=split,test_evaluated=split=="test",metadata=manifest)
        write_json(path,report)
    values[4]=signature
    return values


def regional_fixture():
    cache,pool=region_fixture((1,2,0,1),np.float64)
    cache["area"]=np.array([5,7,5,8],dtype=np.int64)
    cache["relative_area"]=cache["area"]/1000.
    rng=np.random.default_rng(2026);sources={}
    allpoints=[];coords=[];keeps=[]
    for token,n in enumerate(cache["area"]):
        values=rng.random((n,3));values[-1]=values[0]
        allpoints.append(values);coords.append(np.column_stack((np.arange(n)//10,np.arange(n)%10)))
        keeps.append(np.arange(0,n,2) if token>1 else np.arange(token))
    for mode in ("all_points","wbbs_denoised"):
        indices=[np.arange(n) for n in cache["area"]] if mode=="all_points" else keeps
        sources[mode]={"offsets":np.r_[0,np.cumsum([len(i) for i in indices])],
            "descriptors":np.concatenate([v[i] for v,i in zip(allpoints,indices)]),
            "coordinates":np.concatenate([v[i] for v,i in zip(coords,indices)])}
    cache["point_sets"]=sources
    provider=scoring.Provider(cache,pool)
    labels=np.zeros((int(max(pool.gallery))+1,8),dtype=int);classes=np.zeros(len(labels),dtype=int)
    for i,q in enumerate(pool.gallery):labels[q,i%3]=1;classes[q]=i%3
    return provider,SimpleNamespace(labels=labels,class_ids=classes)


class RegionalFormulaTests(unittest.TestCase):
    def test_independent_grid_equals_declared_grid(self):
        self.assertEqual(verify.independent_grid(),state.grid())

    def test_original_bbp_empty_singleton_duplicate_and_all_thirteen_fusions(self):
        provider,bundle=regional_fixture()
        receipt=verify.audit_formulas(provider,bundle,state.grid())
        self.assertEqual(receipt["candidate_pairs"],3*3*13)
        self.assertTrue(receipt["bb_counts_exact"])
        self.assertGreater(receipt["bb_region_pairs_checked"],0)
        self.assertLessEqual(receipt["max_wbbs_absolute_error"],1e-15)
        self.assertLessEqual(receipt["max_matching_absolute_error"],1e-12)

    def test_corrupted_integer_counts_and_count_normalization_are_detected(self):
        for issue in ("count","score"):
            with self.subTest(issue=issue):
                provider,bundle=regional_fixture();original=provider.parts
                def changed(q):
                    parts=original(q)
                    if len(provider.tokens(q)):
                        token=provider.tokens(provider.pool[q][0])[0]
                        if issue=="count":parts["bb_counts"]["all_points"][0,token]+=1
                        else:parts["wbbs_scores"]["all_points"][0.][0,token]+=.01
                    return parts
                with patch.object(provider,"parts",side_effect=changed):
                    with self.assertRaisesRegex(ValueError,"BBP|normalization"):
                        verify.audit_formulas(provider,bundle,state.grid())

    def test_matching_uses_full_query_area_denominator(self):
        provider,bundle=regional_fixture();original=provider.score
        def changed(*args):
            ids,scores=original(*args);return ids,scores+.01
        with patch.object(provider,"score",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"greedy"):
                verify.audit_formulas(provider,bundle,{"all_bb_a04":state.grid()["all_bb_a04"]})


class CacheVerificationTests(unittest.TestCase):
    def test_all_four_stages_recompute_metrics_and_complete_pool_denominators(self):
        for stage in ("valid","cross_valid","test","cross_test"):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as root:
                values=fixture(root,stage)
                receipt=verify.verify_evaluation(*values)
                self.assertEqual(receipt["stage"],stage)
                self.assertEqual(receipt["split"],"valid" if stage.endswith("valid") else "test")
                self.assertEqual(receipt["full_candidates_checked"],sum(len(values[1][q]) for q in values[1]))
                self.assertEqual(receipt["metric_query_counts"]["coverage"],receipt["candidate_protocol_audit"]["metric_query_counts"])
                if stage.startswith("cross_"):
                    self.assertFalse(receipt["exact_reports"])
                    self.assertEqual(receipt["metric_query_counts"]["coverage"]["Coverage-nDCG@1"],12)
                else:self.assertTrue(receipt["exact_reports"])

    def test_global_wbbs_is_only_an_extra_cross_valid_benchmark(self):
        for stage in ("cross_valid","valid","test","cross_test"):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as root:
                values=fixture(root,stage,"global_wbbs")
                if stage=="cross_valid":self.assertEqual(verify.verify_evaluation(*values)["variant"],"global_wbbs")
                else:
                    with self.assertRaisesRegex(ValueError,"frozen specification"):verify.verify_evaluation(*values)

    def test_complete_tie_break_saved_scores_and_unsaved_candidates(self):
        for issue in ("order","score","unsaved"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root);folder,pool,_,_,_,provider,_=values
                if issue=="order":
                    def swap(rows):rows[0]["candidate_id"],rows[1]["candidate_id"]=rows[1]["candidate_id"],rows[0]["candidate_id"]
                    rewrite_csv(folder/"rankings.csv",swap)
                elif issue=="score":rewrite_csv(folder/"rankings.csv",lambda rows:rows[0].update(similarity_score=.123))
                else:
                    q=int(next(iter(pool)))
                    with (folder/"rankings.csv").open() as f:ids={int(r["candidate_id"]) for r in csv.DictReader(f) if int(r["query_id"])==q}
                    missing=next(int(c) for c in pool[q] if int(c) not in ids);provider.values[q,missing]=1.
                with self.assertRaises(ValueError):verify.verify_evaluation(*values)

    def test_full_denominators_nan_counts_and_groups_are_checked_for_cross_valid(self):
        for issue in ("denominator","undefined","group","count"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root);folder=values[0]
                if issue=="denominator":rewrite_csv(folder/"coverage_per_query.csv",lambda rows:rows[0].update(full_coverage_in_pool=123))
                elif issue=="undefined":
                    def change(rows):next(r for r in rows if r["Coverage-nDCG@1"]=="")["Coverage-nDCG@1"]=0.
                    rewrite_csv(folder/"coverage_per_query.csv",change)
                else:
                    path=folder/"coverage_metrics.json";report=verify._json(path)
                    if issue=="count":report["metric_query_counts"]["FullCoverage_mAP@1"]+=1
                    else:report["groups"]["mixed_label"]["num_queries"]+=1
                    write_json(path,report)
                with self.assertRaises(ValueError):verify.verify_evaluation(*values)

    def test_source_cache_binding_and_self_consistent_wrong_spec_are_rejected(self):
        for issue in ("binding","marker","spec","dtype"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root,"cross_valid","all_wbbs_a04");folder=values[0]
                if issue=="binding":values[6].clear()
                elif issue=="marker":write_json(values[5].marker,{"changed":True})
                elif issue=="dtype":
                    original=values[5].scores;values[5].scores=lambda q,c:original(q,c).astype(np.float32)
                else:
                    m=verify._json(folder/"evaluation_manifest.json");m["spec"]["count_alpha"]=7.
                    sig=state.evaluation_signature(m["context_signature"],m["stage"],m["variant"],m["spec"])
                    m["signature"]=m["evaluation_signature"]=sig;values[4]=sig
                    ident=verify._json(folder/"run_identity.json");ident["evaluation_signature"]=sig
                    write_json(folder/"evaluation_manifest.json",m);write_json(folder/"run_identity.json",ident)
                with self.assertRaises(ValueError):verify.verify_evaluation(*values)


if __name__=="__main__":unittest.main()
