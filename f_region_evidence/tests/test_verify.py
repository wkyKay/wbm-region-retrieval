"""Corruption checks for independent formulas and complete-cache replay."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f_region_evidence import calibration, scoring, state, verify
from f_region_evidence.tests.test_scoring import fixture as region_fixture
from f_region_structure.tests.test_verify import fixture as ordinary_fixture
from f_region_j_cross_test.tests.test_verify import fixture as cross_fixture
from fgw_cross_test.tests.test_verify import rewrite_csv, write_json


def fixture(root, *, cross=False):
    root = Path(root)
    if cross:
        folder, pool, bundle, ks, _, provider, binding = cross_fixture(root)
        stage = "cross_test"
    else:
        folder = root / "evaluation"; folder.mkdir()
        pool, bundle, ks, _, replay, matrix = ordinary_fixture(folder)
        provider = SimpleNamespace(signature="cache-signature", marker=root/"cache_completed.json",
            protocol_signature=pool.protocol_signature, query_ids=np.asarray(list(pool)), gallery_ids=pool.gallery)
        provider.scores = lambda q, ids: np.array([matrix[int(q), int(c)] for c in ids], dtype=np.float64)
        provider.values = matrix
        write_json(provider.marker, {"signature":provider.signature})
        source = root/"source.json"; write_json(source,{"frozen":True})
        binding = {str(source.resolve()):verify._sha(source)}
        stage = "valid"
    name, context = "base4_none", "context-signature"
    spec = state.grid()[name]
    signature = state.evaluation_signature(context, stage, name)
    identity = dict(key="f_region_evidence_fixture_"+stage,run_id="20261120T210203+0800__fixture_"+stage,
                    started_at="2026-11-20T21:02:03+08:00",evaluation_signature=signature,
                    variant=name,split=pool.split_name,stage=stage,
                    protocol=verify.PROTOCOL if cross else "retrieval_v1")
    manifest = dict(**identity,signature=signature,context_signature=context,spec=spec,ks=ks,tie_seed=2026,
                    num_queries=len(pool),gallery_count=len(pool.gallery),
                    candidate_rule=verify.RULE if cross else "all_gallery_except_query",
                    retrieval_protocol_signature=pool.protocol_signature,
                    provider_source={"signature":provider.signature,"completed_sha256":verify._sha(provider.marker)},
                    source_binding_sha256=verify._digest(binding),formal_protocol=False)
    if cross:
        manifest["derived_protocol"] = pool.metadata
    write_json(folder/"evaluation_manifest.json",manifest)
    write_json(folder/"run_identity.json",identity)
    for filename in (["coverage_metrics.json"] if cross else ["metrics.json","coverage_metrics.json"]):
        path=folder/filename; value=verify._json(path); value["metadata"]=manifest; write_json(path,value)
    return folder,pool,bundle,ks,signature,provider,binding


def region_providers(dtype=np.float64):
    cache,pool=region_fixture((3,4,0,1,2),dtype)
    train,_=region_fixture((2,4,1,3,2),dtype,start=100)
    reference=calibration.select_reference(train,size=4)
    thresholds,_=calibration.calibrate(cache,train,reference)
    provider=scoring.Provider(cache,pool,thresholds=thresholds)
    labels=np.zeros((max(pool.gallery)+1,8),dtype=int)
    classes=np.zeros(len(labels),dtype=int)
    for i,q in enumerate(pool.gallery):
        labels[q,i%3]=1;classes[q]=i%3
    bundle=SimpleNamespace(labels=labels,class_ids=classes)
    return {"base4":provider},bundle,train,reference


class EvidenceFormulaTests(unittest.TestCase):
    def test_independent_literals_equal_frozen_grid(self):
        self.assertEqual(verify.independent_grid(),state.grid())

    def test_original_regions_are_checked_at_every_preserved_row(self):
        old,_=region_fixture((2,4,0,1,2),np.float32)
        new,_=region_fixture((3,8,1,1,4),np.float32)
        mapping=[]
        for i in range(len(old["map_ids"])):
            a,b=old["offsets"][i:i+2];c=new["offsets"][i]
            mapping.extend(range(c,c+b-a))
        mapping=np.asarray(mapping)
        for key in old:
            if key not in ("map_ids","offsets"):
                new[key][mapping]=old[key]
        new["base_token_indices"]=mapping
        new["base_region_counts"]=np.diff(old["offsets"])
        receipt=verify.audit_preserved_regions(old,new)
        self.assertEqual(receipt["original_regions"],9)
        for key in ("features","joint_all","area","histograms"):
            with self.subTest(key=key):
                changed=deepcopy(new)
                changed[key][mapping[-1]] += 1
                with self.assertRaisesRegex(ValueError,"field changed"):
                    verify.audit_preserved_regions(old,changed)
        changed=deepcopy(new);changed["features"]=changed["features"].astype(np.float64)
        with self.assertRaisesRegex(ValueError,"field changed"):verify.audit_preserved_regions(old,changed)
        changed=deepcopy(new);changed["base_token_indices"]=mapping[::-1]
        with self.assertRaisesRegex(ValueError,"mapping"):verify.audit_preserved_regions(old,changed)

    def test_rejection_tau_one_and_unmatched_region_denominator(self):
        raw=np.array([[.9,.8],[.4,.3],[1.,.2]])
        net=verify.independent_evidence(raw,np.array([.5,.5,1.]))
        np.testing.assert_allclose(net,np.array([[.8,.6],[0.,0.],[0.,0.]]),atol=2e-16,rtol=0.)
        # Only one region matches; all three still contribute to the denominator.
        self.assertAlmostEqual(verify.independent_greedy(net,[1,4,9]),.8/6)
        self.assertEqual(verify.independent_greedy(np.zeros((0,2)),[]),0.)

    def test_formula_audit_all_rejections_and_empty_maps(self):
        for dtype in (np.float32,np.float64):
            providers,bundle,_,_=region_providers(dtype)
            specs={name:spec for name,spec in state.grid().items() if spec["proposal"]=="base4"}
            receipt=verify.audit_formulas(providers,bundle,specs)
            self.assertEqual(set(receipt["variants"]),set(specs))
            self.assertEqual(receipt["candidate_pairs"],3*4*4)
            self.assertLessEqual(receipt["max_raw_formula_absolute_error"],2e-6)
            self.assertLessEqual(receipt["max_matching_absolute_error"],1e-12)

    def test_formula_corruption_and_matching_corruption_are_rejected(self):
        providers,bundle,_,_=region_providers()
        provider=providers["base4"]; original=provider.parts
        def broken(q):
            value=original(q);value["cosine"]=np.minimum(value["cosine"]+.05,1.);return value
        with patch.object(provider,"parts",side_effect=broken):
            with self.assertRaisesRegex(ValueError,"formula mismatch"):
                verify.audit_formulas(providers,bundle,{"base4_none":state.grid()["base4_none"]})
        original_score=provider.score
        def broken_score(*args):
            ids,scores=original_score(*args);return ids,scores+.01
        with patch.object(provider,"score",side_effect=broken_score):
            with self.assertRaisesRegex(ValueError,"aggregation mismatch"):
                verify.audit_formulas(providers,bundle,{"base4_none":state.grid()["base4_none"]})

    def test_train_reference_quantiles_and_sampling_identity(self):
        providers,bundle,train,reference=region_providers(np.float32)
        with patch.object(calibration,"calibrate",side_effect=AssertionError("production calibration")), \
             patch.object(calibration,"select_reference",side_effect=AssertionError("production sampling")), \
             patch.object(scoring.RegionPairScorer,"scores",side_effect=AssertionError("production formula")):
            receipt=verify.audit_calibration(providers,bundle,train,reference)
        self.assertEqual(receipt["reference_size"],4)
        self.assertLessEqual(receipt["max_threshold_absolute_error"],2e-6)
        changed=deepcopy(reference);changed["token_indices"][0]+=1
        with self.assertRaisesRegex(ValueError,"selection identity"):
            verify.audit_calibration(providers,bundle,train,changed)
        provider=providers["base4"]
        provider.thresholds["q75"][provider.tokens(min(provider.pool.gallery))[0]]+=.01
        with self.assertRaisesRegex(ValueError,"linear quantile"):
            verify.audit_calibration(providers,bundle,train,reference)


class EvidenceCacheVerificationTests(unittest.TestCase):
    def test_ordinary_and_cross_full_replay_and_counts(self):
        for cross in (False,True):
            with self.subTest(cross=cross),tempfile.TemporaryDirectory() as root:
                values=fixture(root,cross=cross)
                with patch.object(scoring,"score_components",side_effect=AssertionError("no new scoring")):
                    receipt=verify.verify_evaluation(*values)
                self.assertEqual(receipt["exact_reports"],not cross)
                self.assertEqual(receipt["full_candidates_checked"],sum(len(values[1][q]) for q in values[1]))
                self.assertEqual(set(receipt["files"]),set(verify.CROSS_REPORT_FILES if cross else verify.REPORT_FILES))
                self.assertEqual(set(receipt["metric_query_counts"]),{"coverage"} if cross else {"exact","coverage"})
                self.assertLess(receipt["max_metric_absolute_error"],1e-12)

    def test_unreturned_high_scoring_candidate_is_detected(self):
        for cross in (False,True):
            with self.subTest(cross=cross),tempfile.TemporaryDirectory() as root:
                values=fixture(root,cross=cross);folder,pool,_,_,_,provider,_=values
                query=int(next(iter(pool)))
                with (folder/"rankings.csv").open() as f:
                    selected={int(r["candidate_id"]) for r in csv.DictReader(f) if int(r["query_id"])==query}
                candidate=next(int(c) for c in pool[query] if int(c) not in selected)
                provider.values[query,candidate]=1.
                with self.assertRaisesRegex(ValueError,"full-candidate"):
                    verify.verify_evaluation(*values)

    def test_frozen_formula_self_consistent_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values=list(fixture(root));folder=values[0]
            m=verify._json(folder/"evaluation_manifest.json");m["spec"]["formula"]="C"
            sig=verify._digest({"context":m["context_signature"],"stage":m["stage"],"variant":m["variant"],"spec":m["spec"]})
            m["signature"]=m["evaluation_signature"]=sig;values[4]=sig
            ident=verify._json(folder/"run_identity.json");ident["evaluation_signature"]=sig
            write_json(folder/"evaluation_manifest.json",m);write_json(folder/"run_identity.json",ident)
            with self.assertRaisesRegex(ValueError,"frozen specification"):verify.verify_evaluation(*values)

    def test_source_binding_cache_identity_dtype_and_values(self):
        for issue in ("binding","marker","query_order","protocol","float32","nan"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root);provider=values[5]
                if issue=="binding":values[6].clear()
                elif issue=="marker":write_json(provider.marker,{"changed":True})
                elif issue=="query_order":provider.query_ids=provider.query_ids[::-1]
                elif issue=="protocol":provider.protocol_signature="changed"
                elif issue=="float32":
                    original=provider.scores;provider.scores=lambda q,c:original(q,c).astype(np.float32)
                else:provider.values[next(iter(provider.values))]=float("nan")
                with self.assertRaises(ValueError):verify.verify_evaluation(*values)

    def test_report_order_denominators_groups_and_metadata_are_checked(self):
        for issue in ("order","denominator","undefined","group","metadata"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root,cross=True);folder=values[0]
                if issue=="order":rewrite_csv(folder/"rankings.csv",lambda rows:rows.__setitem__(slice(0,10),rows[5:10]+rows[:5]))
                elif issue=="denominator":rewrite_csv(folder/"coverage_per_query.csv",lambda rows:rows[0].update(full_coverage_in_pool=999))
                elif issue=="undefined":
                    def change(rows):
                        row=next(r for r in rows if r["Coverage-nDCG@1"]=="");row["Coverage-nDCG@1"]=0
                    rewrite_csv(folder/"coverage_per_query.csv",change)
                else:
                    p=folder/"coverage_metrics.json";report=verify._json(p)
                    if issue=="group":report["groups"]["single_label"]["num_queries"]+=1
                    else:report["metadata"]["source_binding_sha256"]="changed"
                    write_json(p,report)
                with self.assertRaises(ValueError):verify.verify_evaluation(*values)

    def test_provider_marker_mutation_during_replay_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            values=fixture(root);provider=values[5];original=provider.scores
            def changed(q,c):
                write_json(provider.marker,{"changed":True});return original(q,c)
            provider.scores=changed
            with self.assertRaisesRegex(ValueError,"changed during verification"):verify.verify_evaluation(*values)


if __name__=="__main__":unittest.main()
