"""Registration admission checks and preservation of already-published records."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from f_region_wbbs import register, protocol, state, verify
from wbm_retrieval.common import digest, file_hash, read_json, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def reseal(folder,signature):
    (folder/"completed.json").unlink(missing_ok=True);state.seal(folder,signature)


def fixture(root,stage="valid",name="all_wbbs_a04"):
    output=root/"artificial/wbbs_synthetic";output.mkdir(parents=True)
    split="valid" if stage.endswith("valid") else "test";cross=stage.startswith("cross_")
    f=protocol.FROZEN[split];nq,ng=f["num_queries"],f["gallery_count"]
    ctx="synthetic-context";spec=state.grid()[name];sig=state.evaluation_signature(ctx,stage,name,spec)
    source=root/"source.json";write_json(source,{"synthetic":True});binding={str(source):file_hash(source)}
    chosen={"winner":name,"best_regional_wbbs":name,"frozen_at":"2026-11-21T00:10:00+08:00"}
    write_json(output/"selection.json",chosen)
    for p in (output/"cache"/split,output/"counts"/split):
        p.mkdir(parents=True);write_json(p/"completed.json",{"synthetic":True})
    scores=output/"score_cache"/split/name;scores.mkdir(parents=True)
    cs=digest({"context":ctx,"split":split,"variant":name,"spec":spec,"score_cache_version":1})
    for filename in ("scores.npy","identities.npz"):(scores/filename).write_bytes(b"synthetic receipt-only fixture")
    write_json(scores/"metadata.json",{"signature":cs,"context_signature":ctx,"split":split,"variant":name,
        "spec":spec,"retrieval_protocol_signature":protocol.BASE_PROTOCOL_SIGNATURE,"num_queries":nq,"gallery_count":ng,
        "dtype":"float64","self_scores":"zero; never ranked","source_cache_sha256":file_hash(output/"cache"/split/"completed.json"),
        "counts_sha256":file_hash(output/"counts"/split/"completed.json")});state.seal(scores,cs)
    provider={"signature":cs,"completed_sha256":file_hash(scores/"completed.json")}
    folder=output/stage/name;folder.mkdir(parents=True);key="f_region_wbbs_synthetic_"+stage+"_"+name
    ident={"schema_version":1,"key":key,"run_id":"20261121T002000+0800__"+key,
        "started_at":"2026-11-21T00:20:00+08:00","evaluation_signature":sig,"variant":name,
        "split":split,"stage":stage,"protocol":protocol.PROTOCOL if cross else "retrieval_v1"}
    candidate_hash=f["candidate_identity_sha256"] if cross else "a"*64
    audit={"candidate_rule":protocol.RULE if cross else "all_gallery_except_query",
        "retrieval_protocol_signature":f["derived_signature"] if cross else protocol.BASE_PROTOCOL_SIGNATURE,
        "candidate_identity_sha256":candidate_hash,"candidate_count_min":f["candidate_count_min"] if cross else ng-1,
        "candidate_count_max":f["candidate_count_max"] if cross else ng-1,
        "query_ids_sha256":f["query_ids_sha256"],"gallery_ids_sha256":f["gallery_ids_sha256"],
        "full_candidates_checked":f["candidate_pairs"] if cross else nq*(ng-1),
        "metric_query_counts":{f"{stem}@{k}":(f["ndcg_valid"] if stem=="Coverage-nDCG" else
            f["full_valid"] if stem in ("FullCoverageRecall","FullCoverage_mAP") else nq) if cross else nq
            for stem in protocol.STEMS for k in state.KS}}
    manifest={**ident,"signature":sig,"context_signature":ctx,"spec":spec,"ks":state.KS,"tie_seed":2026,
        "num_queries":nq,"gallery_count":ng,"candidate_rule":audit["candidate_rule"],
        "retrieval_protocol_signature":audit["retrieval_protocol_signature"],"provider_source":provider,
        "source_binding_sha256":digest(binding),"formal_protocol":True,"training_required":False,
        "posthoc_research":True,"test_parameter_search":False,"selection_split":"valid","selection_protocol":protocol.PROTOCOL,
        "primary_variant":name if split=="test" else None,"best_regional_wbbs":name if split=="test" else None,
        "frozen_selection_sha256":file_hash(output/"selection.json") if split=="test" else None}
    if cross:
        manifest["derived_protocol"]={"protocol_version":protocol.PROTOCOL,"candidate_rule":protocol.RULE,
            "base_protocol_signature":protocol.BASE_PROTOCOL_SIGNATURE,"candidate_identity_sha256":candidate_hash,
            "filter_uses_ground_truth":True,"filter_scope":"benchmark eligibility only; method scores never receive labels",
            "candidate_count_min":f["candidate_count_min"],"candidate_count_max":f["candidate_count_max"]}
    write_json(folder/"evaluation_manifest.json",manifest);write_json(folder/"run_identity.json",ident)
    counts={};entry={};hits=nq*5
    for family in (["coverage"] if cross else ["exact","coverage"]):
        metrics={f"{stem}@{k}":.5 for stem in (("Hit","Precision","Recall","mAP") if family=="exact" else protocol.STEMS) for k in state.KS}
        counts[family]=audit["metric_query_counts"] if family=="coverage" else {m:nq for m in metrics}
        stem="metrics" if family=="exact" else "coverage_metrics"
        write_json(folder/(stem+".json"),{"split":split,"num_queries":nq,"num_classes":37,"ks":state.KS,
            "metric_version":family+"_v1","test_evaluated":split=="test","metadata":manifest,"metrics":metrics,
            "metric_query_counts":counts[family]});entry[stem]=metrics
    files=verify.CROSS_REPORT_FILES if cross else verify.REPORT_FILES
    for filename in files:
        if not (folder/filename).exists():(folder/filename).write_text("synthetic receipt fixture\n")
    receipt={"status":"verified","signature":sig,"context_signature":ctx,"stage":stage,"split":split,"variant":name,
        "spec":spec,"ks":state.KS,"tie_seed":2026,"num_queries":nq,"ranking_rows":nq*500,
        "full_candidates_checked":audit["full_candidates_checked"],"all_queries_full_candidate_topk_exact":True,
        "exact_reports":not cross,"provider_source":provider,"source_binding_sha256":digest(binding),"source_file_count":len(binding),
        "aggregation_values_checked":100,"files":{f:file_hash(folder/f) for f in files},"ranking_sha256":file_hash(folder/"rankings.csv"),
        "max_metric_absolute_error":0.,"max_saved_score_absolute_error":0.,"metric_query_counts":counts,
        "candidate_protocol_audit":audit,**{k:audit[k] for k in ("candidate_count_min","candidate_count_max","candidate_identity_sha256","query_ids_sha256","gallery_ids_sha256")}}
    if not cross:receipt["top10_hits"]=hits;entry["top10_hits"]=hits
    write_json(folder/"verification.json",receipt);state.seal(folder,sig);entry["completed_sha256"]=file_hash(folder/"completed.json")
    values=[output,root,stage,name,spec,ctx,binding,chosen,{"results":{name:entry}},{},audit]
    refresh(values);return values


def refresh(v):
    output,_,stage,name,spec,ctx,_,_,index,files,_=v;folder=output/stage/name
    reseal(folder,state.evaluation_signature(ctx,stage,name,spec));index["results"][name]["completed_sha256"]=file_hash(folder/"completed.json")
    files.clear();files.update({str(p.relative_to(output)):file_hash(p) for p in output.rglob("*") if p.is_file()})


class AdmissionTests(unittest.TestCase):
    def test_four_stages_keep_distinct_protocols_and_correct_families(self):
        for stage in ("valid","cross_valid","test","cross_test"):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                values=fixture(Path(tmp)/"wbm-region-retrieval",stage);result=register._evaluation(*values)
                self.assertEqual(result["split"],"valid" if stage.endswith("valid") else "test")
                self.assertEqual(result["required_families"],["coverage"] if stage.startswith("cross_") else ["exact","coverage"])

    def test_self_consistently_resealed_incomplete_receipt_is_rejected(self):
        cases={"full_candidates_checked":1,"all_queries_full_candidate_topk_exact":False,"files":{},
            "max_metric_absolute_error":float("nan"),"metric_query_counts":{},"candidate_protocol_audit":{},"provider_source":{}}
        for field,value in cases.items():
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                v=fixture(Path(tmp)/"wbm-region-retrieval","cross_valid");path=v[0]/v[2]/v[3]/"verification.json"
                r=read_json(path);r[field]=value;path.write_text(json.dumps(r));refresh(v)
                with self.assertRaises(ValueError):register._evaluation(*v)

    def test_cross_valid_zero_denominators_cannot_be_relabelled_as_test_or_all500(self):
        for count in (500,7000):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as tmp:
                v=fixture(Path(tmp)/"wbm-region-retrieval","cross_valid");folder=v[0]/v[2]/v[3]
                report=read_json(folder/"coverage_metrics.json");report["metric_query_counts"]["Coverage-nDCG@10"]=count
                write_json(folder/"coverage_metrics.json",report);receipt=read_json(folder/"verification.json")
                receipt["metric_query_counts"]["coverage"]=report["metric_query_counts"]
                receipt["files"]["coverage_metrics.json"]=file_hash(folder/"coverage_metrics.json")
                write_json(folder/"verification.json",receipt);refresh(v)
                with self.assertRaisesRegex(ValueError,"effective-query"):register._evaluation(*v)

    def test_missing_cache_source_or_prior_valid_freeze_is_rejected(self):
        for issue in ("cache","time","source"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as tmp:
                v=fixture(Path(tmp)/"wbm-region-retrieval","test")
                if issue=="cache":v[9].pop("score_cache/test/"+v[3]+"/scores.npy")
                elif issue=="time":v[7]["frozen_at"]="2026-11-21T00:30:00+08:00"
                else:
                    Path(next(iter(v[6]))).write_text("changed")
                    with self.assertRaisesRegex(ValueError,"source changed"):register._bound_files(v[6])
                    continue
                with self.assertRaises(ValueError):register._evaluation(*v)

    def test_unsealed_incomplete_design_cannot_be_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"artificial/synthetic";output.mkdir(parents=True)
            write_json(output/"run_manifest.json",state.signed({"version":state.VERSION,"context":{}}))
            with self.assertRaises(ValueError):register.proposal(output,root=Path(tmp))

    def test_original_j_control_requires_complete_replay_and_bound_historical_bytes(self):
        for issue in (None,"partial","cache","source"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as tmp:
                v=fixture(Path(tmp)/"wbm-region-retrieval","valid","region_j")
                source=v[1]/"historical/rankings.csv";source.parent.mkdir();source.write_text("frozen historical ranks")
                v[6][str(source)]=file_hash(source)
                receipt={"status":"verified","split":"valid","num_queries":500,"ranking_rows":250000,
                    "top500_identical":True,"max_score_error":0.,"source":str(source),"source_sha256":file_hash(source),
                    "score_cache_sha256":file_hash(v[0]/"score_cache/valid/region_j/completed.json")}
                if issue=="partial":receipt["ranking_rows"]=500
                elif issue=="cache":receipt["score_cache_sha256"]="a"*64
                elif issue=="source":v[6].pop(str(source))
                write_json(v[0]/"baseline_replay_valid.json",receipt)
                if issue is None:self.assertEqual(register._baseline(v[0],"valid",v[5],v[6],v[9]),receipt)
                else:
                    with self.assertRaisesRegex(ValueError,"baseline"):register._baseline(v[0],"valid",v[5],v[6],v[9])

    def test_self_consistent_partial_or_nonexact_formula_audit_is_rejected(self):
        for issue in ("count","bb","queries","variant","nonfinite"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp);specs=state.grid();queries=list(range(37))
                row={"query_ids":queries,"candidate_pairs":740,"max_raw_formula_absolute_error":0.,"max_matching_absolute_error":0.}
                formula={**row,"status":"verified","num_queries":37,"candidate_pairs":740*len(specs),
                    "bb_region_pairs_checked":100,"bb_counts_exact":True,"max_wbbs_absolute_error":0.,
                    "variants":{name:deepcopy(row) for name in specs}}
                if issue=="count":formula["candidate_pairs"]-=1
                elif issue=="bb":formula["bb_counts_exact"]=False
                elif issue=="queries":formula["query_ids"]=list(range(36))
                elif issue=="variant":formula["variants"]["region_j"]["candidate_pairs"]=1
                else:formula["max_wbbs_absolute_error"]=float("nan")
                (output/"formula_audit_valid.json").write_text(json.dumps(formula))
                with self.assertRaises(ValueError):register._audits(output,"valid",specs,"ctx",{}, {})


class FrozenProtocolAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        with (root/"protocol/deduplicated/split.csv").open() as stream:rows=list(csv.DictReader(stream))
        labels=np.zeros((max(int(r["sample_id"]) for r in rows)+1,8),dtype=np.uint8)
        for r in rows:labels[int(r["sample_id"])]=[int(r[f"label_{i}"]) for i in range(8)]
        def ids(name):
            with (root/"protocol/retrieval_v1"/name).open() as stream:return np.asarray([int(r["sample_id"]) for r in csv.DictReader(stream)])
        base=FullGalleryPool(ids("valid_queries.csv"),ids("valid_gallery.csv"),"valid",protocol.BASE_PROTOCOL_SIGNATURE)
        cls.audits={"valid":protocol.audit_pool(base,labels),"cross_valid":protocol.audit_pool(protocol.make_cross_pool(base,labels),labels)}

    def test_actual_frozen_valid_and_cross_valid_audits_are_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stage,audit in self.audits.items():
                write_json(Path(tmp)/f"protocol_audit_{stage}.json",audit)
                self.assertEqual(register._protocol_audit(Path(tmp),stage),audit)

    def test_cross_valid_cannot_adopt_test_hash_counts_or_changed_groups(self):
        for issue in ("hash","counts","group"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as tmp:
                audit=deepcopy(self.audits["cross_valid"])
                if issue=="hash":audit["candidate_identity_sha256"]=protocol.FROZEN["test"]["candidate_identity_sha256"]
                elif issue=="counts":audit["metric_query_counts"]["Coverage-nDCG@10"]=500
                else:audit["groups"]["mixed_label"]["metric_query_counts"]["FullCoverage_mAP@10"]+=1
                write_json(Path(tmp)/"protocol_audit_cross_valid.json",audit)
                with self.assertRaises(ValueError):register._protocol_audit(Path(tmp),"cross_valid")


class CompatibilityAdmissionTests(unittest.TestCase):
    def fixture(self,root,split="valid"):
        from f_region_wbbs import compat
        output=root/"artificial/synthetic";output.mkdir(parents=True)
        archive=root/compat.ARCHIVED_PATH;archive.parent.mkdir(parents=True)
        archive.write_bytes((compat.ROOT/compat.ARCHIVED_PATH).read_bytes())
        canonical=root/compat.HISTORICAL_PATH;canonical.parent.mkdir(parents=True)
        canonical.write_bytes((compat.ROOT/compat.HISTORICAL_PATH).read_bytes())
        row={"historical_path":compat.HISTORICAL_PATH,"expected_sha256":compat.HISTORICAL_SHA256,
            "archived_path":str(archive.resolve()),"archived_sha256":compat.HISTORICAL_SHA256,
            "current_path":str(canonical.resolve()),"current_sha256":compat.CURRENT_SHA256,"scope":compat.SCOPE}
        binding={str(archive.resolve()):file_hash(archive),str(canonical.resolve()):file_hash(canonical)}
        context={"source_compatibility":[row]};receipt={"context_signature":"ctx","source_compatibility":[row]}
        write_json(output/"selection.json",{"synthetic":"frozen"})
        if split=="test":receipt["selection_sha256"]=file_hash(output/"selection.json")
        write_json(output/f"source_compatibility_{split}.json",receipt)
        write_json(output/f"inputs_{split}.json",context)
        return output,root,split,"ctx",binding,context

    def test_both_real_versions_are_distinct_truthfully_bound_and_metadata_consistent(self):
        for split in ("valid","test"):
            with self.subTest(split=split),tempfile.TemporaryDirectory() as tmp:
                values=self.fixture(Path(tmp),split);rows=register._compatibility(*values)
                self.assertNotEqual(rows[0]["expected_sha256"],rows[0]["current_sha256"])
                self.assertEqual(rows[0]["expected_sha256"],rows[0]["archived_sha256"])

    def test_no_skipped_hash_faked_canonical_hash_or_extra_scientific_mapping(self):
        for issue in ("unbound","archive_bytes","canonical_bytes","fake_hash","science","metadata","test_freeze"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as tmp:
                values=self.fixture(Path(tmp),"test");output,_,split,_,binding,context=values
                path=output/f"source_compatibility_{split}.json";receipt=read_json(path);row=receipt["source_compatibility"][0]
                if issue=="unbound":binding.pop(row["current_path"])
                elif issue=="archive_bytes":Path(row["archived_path"]).write_text("changed historical bytes")
                elif issue=="canonical_bytes":Path(row["current_path"]).write_text("changed current builder")
                elif issue=="fake_hash":row["current_sha256"]=row["expected_sha256"]
                elif issue=="science":row["historical_path"]="f_region_wbbs/scoring.py"
                elif issue=="metadata":context["source_compatibility"]=[]
                else:receipt["selection_sha256"]="a"*64
                write_json(path,receipt)
                with self.assertRaisesRegex(ValueError,"Administrative source compatibility"):register._compatibility(*values)


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
