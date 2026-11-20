"""Synthetic receipt gates; real ranking recomputation lives in test_verify."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_evidence import register, state, verify
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def reseal(folder, signature):
    (folder/"completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root, stage="valid"):
    output=root/"artificial/evidence_synthetic";output.mkdir(parents=True)
    split="valid" if stage=="valid" else "test";cross=stage=="cross_test"
    nq,ng=(500,3599) if split=="valid" else (7199,7199)
    context="synthetic-context";name="base4_q75";spec=state.grid()[name]
    signature=state.evaluation_signature(context,stage,name)
    source=root/"source.json";write_json(source,{"synthetic":True});binding={str(source):file_hash(source)}
    chosen={"winner":name,"family_winners":{"rejection":name},"frozen_at":"2026-11-20T20:00:00+08:00"}
    write_json(output/"selection.json",chosen)
    for p in (output/"cache"/split/"base4",output/"calibration"/split/"base4"):
        p.mkdir(parents=True);write_json(p/"completed.json",{"synthetic":True})
    scores=output/"score_cache"/split/name;scores.mkdir(parents=True)
    cs=digest({"context":context,"split":split,"variant":name,"spec":spec,"score_cache_version":1})
    for filename in ("scores.npy","identities.npz"):(scores/filename).write_bytes(b"synthetic fixture, no actual scores")
    write_json(scores/"metadata.json",{"signature":cs,"context_signature":context,"split":split,
        "variant":name,"spec":spec,"retrieval_protocol_signature":register.catalog.BASE_PROTOCOL_SIGNATURE,
        "num_queries":nq,"gallery_count":ng,"dtype":"float64",
        "source_cache_sha256":file_hash(output/"cache"/split/"base4"/"completed.json"),
        "calibration_sha256":file_hash(output/"calibration"/split/"base4"/"completed.json")})
    state.seal(scores,cs)
    provider={"signature":cs,"completed_sha256":file_hash(scores/"completed.json")}
    folder=output/stage/name;folder.mkdir(parents=True)
    key="f_region_evidence_synthetic_"+stage
    ident={"schema_version":1,"key":key,"run_id":"20261120T210000+0800__"+key,
        "started_at":"2026-11-20T21:00:00+08:00","evaluation_signature":signature,
        "variant":name,"split":split,"stage":stage,"protocol":verify.PROTOCOL if cross else "retrieval_v1"}
    manifest={**ident,"signature":signature,"context_signature":context,"spec":spec,"ks":state.KS,
        "tie_seed":2026,"num_queries":nq,"gallery_count":ng,"candidate_rule":verify.RULE if cross else "all_gallery_except_query",
        "retrieval_protocol_signature":register.catalog.BASE_PROTOCOL_SIGNATURE,"provider_source":provider,
        "source_binding_sha256":digest(binding),"formal_protocol":True,"training_required":False,
        "posthoc_research":True,"test_parameter_search":False,"selection_split":"valid",
        "primary_variant":name if split=="test" else None,
        "family_winners":chosen["family_winners"] if split=="test" else None,
        "frozen_selection_sha256":file_hash(output/"selection.json") if split=="test" else None}
    if cross:
        protocol={"protocol_version":verify.PROTOCOL,"candidate_rule":verify.RULE,
            "base_protocol_signature":register.catalog.BASE_PROTOCOL_SIGNATURE,
            "candidate_identity_sha256":verify._FORMAL_CANDIDATE_HASH}
        manifest["derived_protocol"]=protocol
        manifest["retrieval_protocol_signature"]=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    write_json(folder/"evaluation_manifest.json",manifest);write_json(folder/"run_identity.json",ident)
    counts={};entry={};hits=nq*5
    for family in (["coverage"] if cross else ["exact","coverage"]):
        metrics={f"{metric}@{k}":.5 for metric in
            (("Hit","Precision","Recall","mAP") if family=="exact" else
             ("Coverage-nDCG","MeanCoverage","FullCoverageHit","FullCoveragePrecision","FullCoverageRecall","FullCoverage_mAP")) for k in state.KS}
        counts[family]={m:((7000 if m.startswith("Coverage-nDCG@") else 6200 if m.startswith(("FullCoverageRecall@","FullCoverage_mAP@")) else 7199) if cross else nq) for m in metrics}
        stem="metrics" if family=="exact" else "coverage_metrics"
        write_json(folder/(stem+".json"),{"split":split,"num_queries":nq,"num_classes":37,"ks":state.KS,
            "metric_version":family+"_v1","test_evaluated":split=="test","metadata":manifest,
            "metrics":metrics,"metric_query_counts":counts[family]})
        entry[stem]=metrics
    files=verify.CROSS_REPORT_FILES if cross else verify.REPORT_FILES
    for filename in files:
        if not (folder/filename).exists():(folder/filename).write_text("synthetic receipt only\n")
    receipt={"status":"verified","signature":signature,"context_signature":context,"stage":stage,
        "split":split,"variant":name,"spec":spec,"ks":state.KS,"tie_seed":2026,"num_queries":nq,
        "ranking_rows":nq*500,"full_candidates_checked":50395860 if cross else nq*(ng-1),
        "all_queries_full_candidate_topk_exact":True,"exact_reports":not cross,"provider_source":provider,
        "source_binding_sha256":digest(binding),"source_file_count":len(binding),"aggregation_values_checked":100,
        "files":{f:file_hash(folder/f) for f in files},"ranking_sha256":file_hash(folder/"rankings.csv"),
        "max_metric_absolute_error":0.,"max_saved_score_absolute_error":0.,"metric_query_counts":counts,
        "candidate_count_min":6999 if cross else ng-1,"candidate_count_max":7170 if cross else ng-1,
        "candidate_identity_sha256":verify._FORMAL_CANDIDATE_HASH if cross else "a"*64}
    if not cross:receipt["top10_hits"]=hits;entry["top10_hits"]=hits
    write_json(folder/"verification.json",receipt);state.seal(folder,signature)
    entry["completed_sha256"]=file_hash(folder/"completed.json")
    return output,root,stage,name,context,binding,chosen,{"results":{name:entry}}, {}


def refresh(values):
    output,_,stage,name,context,_,_,index,_=values
    folder=output/stage/name
    reseal(folder,state.evaluation_signature(context,stage,name))
    index["results"][name]["completed_sha256"]=file_hash(folder/"completed.json")
    values[-1].clear();values[-1].update({str(p.relative_to(output)):file_hash(p) for p in output.rglob("*") if p.is_file()})


class EvidenceRegistrationGates(unittest.TestCase):
    def test_ordinary_and_cross_receipts_register_with_distinct_protocols(self):
        for stage in ("valid","test","cross_test"):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"wbm-region-retrieval";values=fixture(root,stage);refresh(values)
                result=register._evaluation(*values)
                self.assertEqual(result["required_families"],["coverage"] if stage=="cross_test" else ["exact","coverage"])
                self.assertFalse(result["repeat_validation_of_existing_result"])

    def test_self_consistently_resealed_incomplete_verification_is_rejected(self):
        for field,value in (("all_queries_full_candidate_topk_exact",False),("full_candidates_checked",10),
                            ("files",{}),("max_metric_absolute_error",float("nan")),
                            ("metric_query_counts",{}),("provider_source",{})):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                values=fixture(Path(tmp)/"wbm-region-retrieval");path=values[0]/"valid"/values[3]/"verification.json"
                receipt=read_json(path);receipt[field]=value
                path.write_text(json.dumps(receipt))
                refresh(values)
                with self.assertRaises(ValueError):register._evaluation(*values)

    def test_changed_report_and_missing_complete_cache_file_are_rejected(self):
        for issue in ("report","cache"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as tmp:
                values=fixture(Path(tmp)/"wbm-region-retrieval");refresh(values)
                if issue=="report":
                    folder=values[0]/"valid"/values[3];report=read_json(folder/"metrics.json")
                    report["metrics"]["Precision@10"]+=.1;write_json(folder/"metrics.json",report);refresh(values)
                else:
                    key="score_cache/valid/"+values[3]+"/scores.npy";values[-1].pop(key)
                with self.assertRaises(ValueError):register._evaluation(*values)

    def test_cross_effective_query_denominators_cannot_be_redefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            values=fixture(Path(tmp)/"wbm-region-retrieval","cross_test")
            folder=values[0]/"cross_test"/values[3]
            report=read_json(folder/"coverage_metrics.json");report["metric_query_counts"]["FullCoverage_mAP@10"]=7199
            write_json(folder/"coverage_metrics.json",report)
            receipt=read_json(folder/"verification.json");receipt["metric_query_counts"]["coverage"]=report["metric_query_counts"]
            receipt["files"]["coverage_metrics.json"]=file_hash(folder/"coverage_metrics.json");write_json(folder/"verification.json",receipt)
            refresh(values)
            with self.assertRaisesRegex(ValueError,"effective-query"):register._evaluation(*values)

    def test_source_bytes_and_selection_time_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            values=fixture(Path(tmp)/"wbm-region-retrieval","test");refresh(values)
            register._bound_files(values[5])
            Path(next(iter(values[5]))).write_text("changed")
            with self.assertRaisesRegex(ValueError,"source changed"):register._bound_files(values[5])
            values[6]["frozen_at"]="2026-11-20T22:00:00+08:00"
            with self.assertRaisesRegex(ValueError,"prior frozen valid"):register._evaluation(*values)

    def test_unsealed_run_cannot_enter_catalogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"artificial/unsealed";output.mkdir(parents=True)
            write_json(output/"run_manifest.json",state.signed({"version":state.VERSION,"context":{}}))
            with self.assertRaises(ValueError):register.proposal(output,root=Path(tmp))


class CatalogPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=(Path(self.tmp.name)/"wbm-region-retrieval").resolve();self.folder=self.root/"results_catalog";self.folder.mkdir(parents=True)
        self.output=self.root/"artificial/synthetic";self.output.mkdir(parents=True)
        self.old={"key":"old","nested":{"metric":.4,"denominator":7199}}
        self.addition={"key":"new","variant":"synthetic"}
        sources={"schema_version":1,"runs":[{"key":"old"}]}
        snapshot={"source_spec_sha256":hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),"runs":[self.old]}
        write_json(self.folder/"sources.json",sources);write_json(self.folder/"snapshot.json",snapshot)
        for name,value in self.render(snapshot).items():(self.folder/name).write_text(value)

    def render(self,snapshot):return {"registry.csv":json.dumps(snapshot["runs"],sort_keys=True)}
    def collect(self,workspace,sources,previous):
        return {"source_spec_sha256":hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                "runs":[deepcopy(self.old),deepcopy(self.addition)]}
    def bytes(self):return {p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file() and not p.name.startswith(".")}
    def mocks(self):
        return (patch.object(register,"proposal",return_value={"schema_version":1,"runs":[self.addition]}),
                patch.object(register.catalog,"collect",side_effect=self.collect),
                patch.object(register.catalog,"render",side_effect=self.render))

    def test_addition_is_idempotent_and_preserves_old_record_deeply(self):
        a,b,c=self.mocks()
        with a,b,c:
            self.assertEqual(register.apply_proposal(self.output,root=self.root)["new_results"],1)
            self.assertEqual(read_json(self.folder/"snapshot.json")["runs"][0],self.old)
            before=self.bytes()
            self.assertEqual(register.apply_proposal(self.output,root=self.root)["status"],"already_registered")
            self.assertEqual(before,self.bytes())

    def test_collection_that_changes_old_nested_value_is_rejected(self):
        before=self.bytes()
        def damaged(*args):
            result=self.collect(*args);result["runs"][0]["nested"]["denominator"]=1;return result
        a,_,c=self.mocks()
        with a,c,patch.object(register.catalog,"collect",side_effect=damaged):
            with self.assertRaisesRegex(ValueError,"prior result"):register.apply_proposal(self.output,root=self.root)
        self.assertEqual(before,self.bytes())

    def test_failed_write_rolls_back_shared_catalogue(self):
        before=self.bytes();original=register._atomic_write;failed=False
        def fail(path,content):
            nonlocal failed
            if path==self.folder/"registry.csv" and not failed:
                failed=True;raise OSError("injected failure")
            return original(path,content)
        a,b,c=self.mocks()
        with a,b,c,patch.object(register,"_atomic_write",side_effect=fail):
            with self.assertRaisesRegex(OSError,"injected"):register.apply_proposal(self.output,root=self.root)
        self.assertEqual(before,self.bytes())


if __name__=="__main__":unittest.main()
