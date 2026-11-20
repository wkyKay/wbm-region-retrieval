"""Coverage-only replay from synthetic complete float64 score caches."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.proposed.f_segmentation_geometry import coverage_metrics
from f_region_area_fusion import scoring as regional_scoring
from fgw_cross_test.tests.test_verify import Pool, rewrite_csv, write_json
from f_region_j_cross_test import verify
from wbm_retrieval.exact_metrics import order_scores


class Cache:
    def __init__(self, root, pool):
        self.signature, self.protocol_signature = "sealed-cache-signature", "base_protocol"
        self.marker = Path(root) / "cache_completed.json"
        write_json(self.marker, {"status":"completed", "signature":self.signature})
        self.query_ids, self.gallery_ids = pool.queries.copy(), pool.gallery.copy()
        q, c = np.meshgrid(self.query_ids, self.gallery_ids, indexing="ij")
        self.values = ((q * 13 + c * 17) % 87 + 6) / 100.
        self.calls = []

    def scores(self, query, candidates):
        self.calls.append(int(query))
        return self.values[int(query), candidates].copy()


def fixture(root, *, ties=False):
    root = Path(root)
    labels = np.asarray([[1,1,1,0,0], [1,1,1,0,0], [1,1,1,1,0], [1,0,1,0,0],
                         [0,1,1,0,0], [0,0,1,0,0], [0,0,0,1,0], [1,0,0,0,0],
                         [0,1,0,0,0], [1,1,0,0,0], [0,0,0,0,1], [1,1,0,0,0],
                         [0,0,1,1,0]], dtype=int)
    labels = np.pad(labels, ((0,0),(0,3)))
    taxonomy = {pattern:i for i,pattern in enumerate(sorted(set(map(tuple,labels))))}
    bundle = SimpleNamespace(labels=labels, class_ids=np.asarray([taxonomy[tuple(row)] for row in labels]))
    pool, ks = Pool(labels), [1,3,5]
    provider = Cache(root,pool)
    if ties:provider.values[:] = .4
    source = root / "upstream_source.json"; write_json(source,{"frozen":"original region features"})
    binding = {str(source.resolve()):verify._sha(source)}
    folder = root / "evaluation"; folder.mkdir()
    context = "context-signature"
    spec = {"family":"add","alpha":.4,"beta":.25}
    variant = "add_a0p4_b0p25"
    signature = verify._digest({"context_signature":context,"variant":variant,"spec":spec})
    identity = dict(run_id="20261120T210203+0800__fixture",started_at="2026-11-20T21:02:03+08:00",
                    evaluation_signature=signature,variant=variant,split="test")
    manifest = dict(**identity,signature=signature,context_signature=context,spec=spec,ks=ks,tie_seed=2026,
                    num_queries=len(pool),gallery_count=len(pool.gallery),candidate_rule=verify.RULE,
                    derived_protocol=pool.metadata,retrieval_protocol_signature=pool.protocol_signature,
                    provider_source={"signature":provider.signature,"completed_sha256":verify._sha(provider.marker)},
                    source_binding_sha256=verify._digest(binding),formal_protocol=False)
    write_json(folder / "run_identity.json",identity)
    write_json(folder / "evaluation_manifest.json",manifest)
    with (folder / "rankings.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(verify.RANKING_FIELDS)
        for q in pool:
            candidates=pool[q];scores=provider.scores(q,candidates)
            order=order_scores(candidates,scores,q,2026)[:max(ks)]
            writer.writerows((q,rank,int(candidates[i]),float(scores[i])) for rank,i in enumerate(order,1))
    coverage_metrics.evaluate_rankings_file(bundle,pool,folder/"rankings.csv",ks,folder,
                                           split_name="test",metadata=manifest)
    provider.calls.clear()
    return folder,pool,bundle,ks,signature,provider,binding


def check(values):
    return verify.verify_evaluation(*values)


class RegionalCrossCacheVerificationTests(unittest.TestCase):
    def test_cache_replay_verifies_all_metrics_without_recomputing_regional_scores(self):
        with tempfile.TemporaryDirectory() as root:
            values=fixture(root)
            with patch.object(coverage_metrics,"query_metrics",side_effect=AssertionError("production metric")), \
                 patch.object(regional_scoring,"score_components",side_effect=AssertionError("regional recomputation")):
                receipt=check(values)
            self.assertEqual(values[5].calls,list(values[1]))
            self.assertEqual(receipt["status"],"verified")
            self.assertEqual(receipt["num_queries"],13)
            self.assertEqual(receipt["ranking_rows"],65)
            self.assertEqual(receipt["full_candidates_checked"],sum(len(values[1][q]) for q in values[1]))
            self.assertEqual(receipt["metric_query_counts"]["Coverage-nDCG@1"],12)
            self.assertEqual(set(receipt["files"]),set(verify.REPORT_FILES))
            self.assertEqual(receipt["max_saved_score_absolute_error"],0.)
            self.assertLess(receipt["max_metric_absolute_error"],1e-12)
            self.assertIn("Not a second regional score computation",receipt["scope"])

    def test_unsaved_candidate_with_higher_cache_score_is_not_missed(self):
        with tempfile.TemporaryDirectory() as root:
            values=fixture(root);folder,pool,_,_,_,provider,_=values
            with (folder/"rankings.csv").open() as stream:
                returned={int(r["candidate_id"]) for r in csv.DictReader(stream) if int(r["query_id"])==0}
            absent=next(int(c) for c in pool[0] if int(c) not in returned)
            provider.values[0,absent]=1.
            with self.assertRaisesRegex(ValueError,"full-candidate cache ordering"):check(values)

    def test_equal_score_tie_order_is_independent_and_strict(self):
        with tempfile.TemporaryDirectory() as root:
            values=fixture(root,ties=True);check(values)
            def swap(rows):
                rows[0]["candidate_id"],rows[1]["candidate_id"]=rows[1]["candidate_id"],rows[0]["candidate_id"]
            rewrite_csv(values[0]/"rankings.csv",swap)
            with self.assertRaisesRegex(ValueError,"cache ordering"):check(values)

    def test_query_groups_must_be_complete_unique_and_in_frozen_order(self):
        for issue in ("swap","duplicate","missing"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root)
                def alter(rows):
                    if issue=="swap":rows[:10]=rows[5:10]+rows[:5]
                    elif issue=="duplicate":rows.extend(rows[:5])
                    else:del rows[-5:]
                rewrite_csv(values[0]/"rankings.csv",alter)
                with self.assertRaisesRegex(ValueError,"query|queries"):check(values)

    def test_excluded_candidate_or_changed_saved_score_is_rejected(self):
        for issue in ("excluded","score"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root)
                rewrite_csv(values[0]/"rankings.csv",lambda rows:rows[0].update(
                    {"candidate_id":1} if issue=="excluded" else {"similarity_score":.12345}))
                with self.assertRaises(ValueError):check(values)

    def test_undefined_metric_cannot_be_zero_and_full_pool_denominator_cannot_change(self):
        for issue in ("undefined","denominator"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root)
                def alter(rows):
                    if issue=="undefined":
                        row=next(r for r in rows if int(r["query_id"])==10)
                        self.assertEqual(row["Coverage-nDCG@1"],"");row["Coverage-nDCG@1"]=0.
                    else:rows[0]["full_coverage_in_pool"]=999
                rewrite_csv(values[0]/"coverage_per_query.csv",alter)
                with self.assertRaisesRegex(ValueError,"Undefined|denominator"):check(values)

    def test_aggregate_effective_counts_groups_macro_and_extra_metric_are_checked(self):
        for issue in ("count","group","macro","extra"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root);path=values[0]/"coverage_metrics.json";report=verify._json(path)
                if issue=="count":report["metric_query_counts"]["FullCoverageRecall@1"]+=1
                if issue=="group":report["groups"]["single_label"]["num_queries"]+=1
                if issue=="macro":report["class_macro_metrics"]["MeanCoverage@1"]+=.01
                if issue=="extra":report["metrics"]["Unrequested@1"]=0.
                write_json(path,report)
                with self.assertRaises(ValueError):check(values)

    def test_source_binding_and_provider_identity_are_strict(self):
        for issue in ("binding","query_order","gallery_order","protocol","marker"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root);provider=values[5]
                if issue=="binding":values[6].clear()
                if issue=="query_order":provider.query_ids=provider.query_ids[::-1]
                if issue=="gallery_order":provider.gallery_ids=provider.gallery_ids[::-1]
                if issue=="protocol":provider.protocol_signature="different"
                if issue=="marker":write_json(provider.marker,{"other":"cache"})
                with self.assertRaises(ValueError):check(values)

    def test_self_consistent_changed_formula_still_fails_independent_literal(self):
        with tempfile.TemporaryDirectory() as root:
            values=list(fixture(root));folder=values[0]
            manifest=verify._json(folder/"evaluation_manifest.json")
            manifest["spec"]["alpha"]=.5
            sig=verify._digest({k:manifest[k] for k in ("context_signature","variant","spec")})
            manifest["signature"]=manifest["evaluation_signature"]=sig
            identity=verify._json(folder/"run_identity.json");identity["evaluation_signature"]=sig
            report=verify._json(folder/"coverage_metrics.json");report["metadata"]=manifest
            write_json(folder/"run_identity.json",identity);write_json(folder/"evaluation_manifest.json",manifest)
            write_json(folder/"coverage_metrics.json",report);values[4]=sig
            with self.assertRaisesRegex(ValueError,"frozen formula"):check(values)

    def test_invalid_cache_values_or_dtype_cannot_be_verified(self):
        for issue in ("float32","nan","negative","shape"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root);provider=values[5]
                if issue=="float32":provider.values=provider.values.astype(np.float32)
                elif issue=="shape":provider.values=provider.values[...,None]
                else:provider.values[:]=np.nan if issue=="nan" else -.1
                with self.assertRaisesRegex(ValueError,"float64"):check(values)

    def test_marker_or_report_mutation_during_read_is_detected(self):
        for target in ("marker","report","binding"):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as root:
                values=fixture(root);provider=values[5];original=provider.scores
                def mutate(q,c):
                    if q==0:
                        if target=="marker":write_json(provider.marker,{"changed":True})
                        elif target=="binding":values[6][str(Path(root)/"extra") ]="0"*64
                        else:
                            path=values[0]/"run_identity.json";data=verify._json(path);data["changed"]=True;write_json(path,data)
                    return original(q,c)
                provider.scores=mutate
                with self.assertRaisesRegex(ValueError,"changed during verification"):check(values)

    def test_extra_exact_report_and_noncanonical_columns_are_rejected(self):
        for issue in ("exact","columns"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as root:
                values=fixture(root)
                if issue=="exact":write_json(values[0]/"metrics.json",{})
                else:
                    path=values[0]/"rankings.csv";path.write_text(path.read_text().replace("similarity_score","wrong_score",1))
                with self.assertRaises(ValueError):check(values)


if __name__=="__main__":
    unittest.main()
