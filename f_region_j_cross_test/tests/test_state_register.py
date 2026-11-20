"""Synthetic completion gates and one-addition catalogue transactions.

Small fixtures exercise provenance only. Full candidate ranking/metrics and
real score cache shape/arithmetic are tested independently by their modules.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from f_region_j_cross_test import state, register
from f_region_area_fusion import scoring, selection
from wbm_retrieval.common import digest, file_hash, read_json, write_json

CODE = {"synthetic_science.py": "a" * 64}


def reseal(folder, signature):
    (folder / "completed.json").unlink(missing_ok=True)
    state.seal(folder, signature)


def fixture(root):
    root = root.resolve()
    valid, test, reference = [root / "artificial" / x for x in ("original_valid", "original_test", "old_cross")]
    derived = dict(protocol_version=state.PROTOCOL, candidate_rule=state.CANDIDATE_RULE,
        base_protocol_signature=state.BASE_PROTOCOL_SIGNATURE, candidate_identity_sha256=state.CANDIDATE_IDENTITY,
        candidate_count_min=6999, candidate_count_max=7170, filter_uses_ground_truth=True)
    vc = {"split": "valid", "version": "f_region_area_fusion_v1"}; vs = digest(vc)
    write_json(valid / "run_manifest.json", state.signed(dict(version="f_region_area_fusion_v1", context=vc)))
    results = {n:dict(split="valid",num_queries=500,top10_hits=2613 if n==state.VARIANT else 2500-i)
               for i,n in enumerate(scoring.frozen_grid())}
    index = dict(context_signature=vs,results=results)
    write_json(valid / "evaluation_index_valid.json",index)
    chosen = state.signed(dict(**selection.select_top10(results),context_signature=vs,
        valid_index_sha256=file_hash(valid / "evaluation_index_valid.json"),frozen_at="2026-11-20T16:04:14+08:00"))
    write_json(valid / "selection.json",chosen); state.seal(valid,vs)
    frozen = state.signed(dict(schema_version=1,source_run=str(valid),selection=chosen,
        selected_variants=chosen["selected_variants"],selected_specs=chosen["selected_specs"],
        selection_policy="frozen_valid_top10_only", source_selection_sha256=file_hash(valid/"selection.json"),
        source_completed_sha256=file_hash(valid/"completed.json"),
        source_valid_index_sha256=file_hash(valid/"evaluation_index_valid.json")))
    tc = dict(split="test",version="f_region_area_fusion_v1",source_run=str(valid),frozen_selection_signature=frozen["signature"])
    ts = digest(tc)
    write_json(test/"run_manifest.json",state.signed(dict(version="f_region_area_fusion_v1",context=tc)))
    write_json(test/"frozen_selection.json",frozen)
    old = test/"test"/state.VARIANT
    oldsig = digest(dict(context=ts,split="test",variant=state.VARIANT,spec=state.SPEC))
    write_json(old/"evaluation_manifest.json",dict(signature=oldsig,variant=state.VARIANT,spec=state.SPEC,
        split="test",matching=state.FACTS["matching"],valid_selection_rank=1,num_queries=7199,gallery_count=7199,
        training_required=False,test_parameter_search=False,retrieval_protocol_signature=state.BASE_PROTOCOL_SIGNATURE))
    (old/"rankings.csv").write_text("synthetic original ranking receipt\n")
    state.seal(old,oldsig);state.seal(test,ts)
    counts = state.expected_metric_counts()
    rc = dict(derived_protocol=derived); rs = digest(rc)
    write_json(reference/"run_manifest.json",state.signed(dict(version="fgj_cross_test_v1",context=rc)))
    folders={}
    for name in state.REFERENCE_VARIANTS:
        folder=reference/"test"/name;folders[name]=str(folder)
        write_json(folder/"coverage_metrics.json",dict(split="test",metric_version="coverage_v1",
            num_queries=7199,ks=state.SETTINGS["ks"],metadata=dict(derived_protocol=derived),
            metric_query_counts=counts,metrics={k:.3 for k in counts}))
        state.seal(folder,"synthetic_reference_"+name)
    state.seal(reference,rs)
    unused = root/"original_feature.npy";unused.write_bytes(b"frozen original feature")
    binding={str(p):file_hash(p) for base in (valid,test,reference) for p in base.rglob('*') if p.is_file()}
    binding[str(unused)]=file_hash(unused)
    metadata=dict(source_valid=str(valid),source_test=str(test),source_winner_folder=str(old),
        reference_root=str(reference),reference_folders=folders,source_selection_signature=chosen["signature"],
        source_valid_context_signature=vs,source_test_context_signature=ts,source_winner_signature=oldsig,
        origin_valid_winner=True,valid_selection_rank=1)
    output=root/"artificial/f_region_j_cross_test_20261120T210000+0800"
    frozen=state.fixed_variants()
    context=dict(**state.FACTS,version=state.VERSION,split="test",code=CODE,settings=deepcopy(state.SETTINGS),
        variant=state.VARIANT,spec=deepcopy(state.SPEC),source_binding=binding,source_metadata=metadata,
        formal_protocol=True,frozen_variants_signature=frozen["signature"],derived_protocol=derived,
        protocol=state.PROTOCOL,candidate_rule=state.CANDIDATE_RULE,retrieval_protocol_signature=state.derived_signature(derived))
    sig=digest(context)
    write_json(output/"run_manifest.json",state.signed(dict(version=state.VERSION,stage="test",split="test",context=context,
        started_at="2026-11-20T21:00:00+08:00",run_id="20261120T210000+0800__f_region_j_cross_test")))
    for name,data in [("frozen_variants.json",frozen),("frozen_selection.json",read_json(test/"frozen_selection.json")),
                      ("settings.json",state.SETTINGS),("candidate_protocol.json",derived),
                      ("inputs.json",dict(source_binding=binding,source_metadata=metadata)),
                      ("source_protection_before.json",{}),("source_protection_after.json",dict(status="unchanged",files=0))]:
        write_json(output/name,data)
    audit=output/"protocol/cross_combination_candidates.csv";audit.parent.mkdir();audit.write_text("synthetic audit\n")
    cache=output/"score_cache";cache.mkdir()
    for name in ("ids.npy","scores.npy"):(cache/name).write_bytes(b"synthetic cache receipt")
    write_json(cache/"metadata.json",dict(context_signature=sig,variant=state.VARIANT,spec=state.SPEC,
        base_protocol_signature=state.BASE_PROTOCOL_SIGNATURE,query_count=7199,gallery_count=7199,
        dtype="float64",self_score="NaN",full_pairs=51818402))
    state.seal(cache,state.cache_signature(sig))
    ps=dict(signature=state.cache_signature(sig),completed_sha256=file_hash(cache/"completed.json"))
    write_json(output/"formula_audit_test.json",dict(status="verified",split="test",context_signature=sig,
        specs={state.VARIANT:state.SPEC},query_ids=list(range(37)),candidate_pairs=740,max_absolute_error=1e-7))
    write_json(output/"historical_replay.json",dict(status="verified",context_signature=sig,num_queries=7199,
        ranking_rows=3599500,top500_identical=True,max_score_error=0.,source=str(old/"rankings.csv"),
        source_sha256=file_hash(old/"rankings.csv"),score_cache_completed_sha256=ps["completed_sha256"]))
    write_json(output/"comparison_references.json",dict(status="verified",context_signature=sig,references={
        n:dict(folder=f,completed_sha256=file_hash(Path(f)/"completed.json"),
               coverage_metrics_sha256=file_hash(Path(f)/"coverage_metrics.json")) for n,f in folders.items()}))
    folder=state.evaluation_folder(output);folder.mkdir(parents=True)
    es=state.variant_signature(sig)
    with patch.object(state,"now",return_value=datetime(2026,11,20,21,0,1,tzinfo=state.TZ)):
        ident=state.identity(output,es)
    ev=dict(**ident,**state.FACTS,signature=es,context_signature=sig,settings=state.SETTINGS,spec=state.SPEC,
        stage="test",protocol=state.PROTOCOL,candidate_rule=state.CANDIDATE_RULE,derived_protocol=derived,
        retrieval_protocol_signature=state.derived_signature(derived),num_queries=7199,gallery_count=7199,
        ks=state.SETTINGS["ks"],tie_seed=2026,provider_source=ps,source_binding_sha256=digest(binding),formal_protocol=True)
    write_json(folder/"run_identity.json",ident);write_json(folder/"evaluation_manifest.json",ev)
    report=dict(split="test",num_queries=7199,ks=state.SETTINGS["ks"],metric_version="coverage_v1",metadata=ev,
        metrics={k:.4 for k in counts},metric_query_counts=counts,queries_without_full_coverage_candidates=999)
    write_json(folder/"coverage_metrics.json",report)
    for name in ("rankings.csv","coverage_per_query.csv","coverage_per_class.csv"):(folder/name).write_text("synthetic report\n")
    files={name:file_hash(folder/name) for name in state.REPORT_FILES}
    receipt=dict(status="verified",signature=es,context_signature=sig,split="test",variant=state.VARIANT,spec=state.SPEC,
        ks=state.SETTINGS["ks"],tie_seed=2026,provider_source=ps,source_binding_sha256=digest(binding),source_file_count=len(binding),
        num_queries=7199,ranking_rows=3599500,full_candidates_checked=50395860,candidate_identity_sha256=state.CANDIDATE_IDENTITY,
        candidate_count_min=6999,candidate_count_max=7170,all_queries_full_candidate_topk_exact=True,exact_reports=False,
        metric_query_counts=counts,ranking_sha256=files["rankings.csv"],files=files,max_metric_absolute_error=0.,
        max_saved_score_absolute_error=0.,scope="Synthetic sealed score cache receipt; not a second regional computation")
    write_json(folder/"verification.json",receipt);write_json(folder/"timing.json",dict(synthetic=True));state.seal(folder,es)
    write_json(output/"evaluation_index_test.json",dict(context_signature=sig,split="test",variant=state.VARIANT,
        spec=state.SPEC,coverage_metrics=report["metrics"],metric_query_counts=counts,completed_sha256=file_hash(folder/"completed.json")))
    write_json(output/"status.json",dict(status="completed",context_signature=sig,test_evaluated=True,new_evaluations=1,
        finished_at="2026-11-20T21:00:02+08:00"))
    state.seal(output,sig)
    return output,sig


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"wbm-region-retrieval"
        self.output,self.signature=fixture(self.root)
        for field in ("_current_code","_check_cache"):
            p=patch.object(state,field,return_value=CODE if field=="_current_code" else None)
            p.start();self.addCleanup(p.stop)

    def mutate(self,name,change):
        path=self.output/name;value=read_json(path);change(value);write_json(path,value);reseal(self.output,self.signature)

    def test_complete_single_role_and_frozen_source_chain(self):
        manifest,frozen=state.verify_run(self.output)
        self.assertEqual(frozen,state.fixed_variants())
        self.assertEqual(digest(manifest["context"]),self.signature)

    def test_identity_is_persistent_and_formula_specific(self):
        sig=state.variant_signature(self.signature);first=state.identity(self.output,sig)
        self.assertEqual(first,state.identity(self.output,sig))
        with self.assertRaises(ValueError):state.identity(self.output,"changed")
        frozen=state.fixed_variants();frozen["specs"][state.VARIANT]["alpha"]=.5
        self.assertEqual(state.fixed_variants()["specs"][state.VARIANT],state.SPEC)

    def test_rejects_seal_using_manifest_instead_of_context(self):
        manifest=state.read_signed(self.output/"run_manifest.json")
        reseal(self.output,manifest["signature"])
        with self.assertRaises(ValueError):state.verify_run(self.output)

    def test_rejects_missing_or_impossible_finish_time(self):
        for value in (None,"2026-11-20T20:59:59+08:00"):
            self.mutate("status.json",lambda x:x.update(finished_at=value))
            with self.assertRaises(ValueError):state.verify_run(self.output)

    def test_rejects_changed_source_and_current_code(self):
        (self.root/"original_feature.npy").write_bytes(b"changed original")
        with self.assertRaises(ValueError):state.verify_run(self.output)
        state.verify_run(self.output,check_source=False)
        with patch.object(state,"_current_code",return_value={}):
            with self.assertRaises(ValueError):state.verify_run(self.output,check_source=False)

    def test_rejects_wrong_source_valid_winner_and_missing_reference(self):
        meta=state.read_signed(self.output/"run_manifest.json")["context"]["source_metadata"]
        selected=Path(meta["source_valid"])/"selection.json";selected.write_text("{}")
        with self.assertRaises(ValueError):state.verify_run(self.output,check_source=False)

    def test_rejects_missing_existing_comparison_reference(self):
        self.mutate("comparison_references.json",lambda x:x["references"].pop("WBBS"))
        with self.assertRaises(ValueError):state.verify_run(self.output)

    def test_rejects_audit_or_historical_replay_drift(self):
        for filename,key,value in [("formula_audit_test.json","query_ids",[0]*37),
                                   ("formula_audit_test.json","max_absolute_error",2.1e-6),
                                   ("historical_replay.json","max_score_error",1e-16),
                                   ("historical_replay.json","score_cache_completed_sha256","wrong")]:
            original=read_json(self.output/filename)
            self.mutate(filename,lambda x:x.update({key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):state.verify_run(self.output)
            write_json(self.output/filename,original);reseal(self.output,self.signature)

    def test_rejects_incomplete_cache_metadata(self):
        cache=self.output/"score_cache"
        metadata=read_json(cache/"metadata.json");metadata["full_pairs"]=50395860
        write_json(cache/"metadata.json",metadata);reseal(cache,state.cache_signature(self.signature));reseal(self.output,self.signature)
        with self.assertRaises(ValueError):state.verify_run(self.output)

    def test_rejects_second_new_endpoint_or_exact_report(self):
        (self.output/"test/J_ONLY").mkdir();reseal(self.output,self.signature)
        with self.assertRaises(ValueError):state.verify_run(self.output)
        (self.output/"test/J_ONLY").rmdir()
        folder=state.evaluation_folder(self.output);write_json(folder/"metrics.json",{})
        reseal(folder,state.variant_signature(self.signature));reseal(self.output,self.signature)
        with self.assertRaises(ValueError):state.verify_run(self.output)

    def test_rejects_receipt_missing_full_candidates_and_wrong_count(self):
        folder=state.evaluation_folder(self.output);receipt=read_json(folder/"verification.json")
        receipt["full_candidates_checked"]=3599500
        write_json(folder/"verification.json",receipt);reseal(folder,state.variant_signature(self.signature));reseal(self.output,self.signature)
        with self.assertRaises(ValueError):state.verify_run(self.output)

    def test_metric_counts_keep_undefined_denominators(self):
        counts=state.expected_metric_counts();self.assertEqual(len(counts),42)
        for k in state.SETTINGS["ks"]:
            self.assertEqual(counts[f"Coverage-nDCG@{k}"],7000)
            self.assertEqual(counts[f"FullCoverage_mAP@{k}"],6200)
            self.assertEqual(counts[f"MeanCoverage@{k}"],7199)

    def test_error_tolerance_rejects_nonfinite_negative_or_boolean(self):
        for value in [float('inf'),float('nan'),-1e-16,True,None]:
            self.assertFalse(state._error(value,2e-6))


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve(); self.root = self.workspace / "wbm-region-retrieval"
        self.output, self.signature = fixture(self.root)
        p = patch.object(state, "_current_code", return_value=CODE); p.start(); self.addCleanup(p.stop)
        p = patch.object(state, "_check_cache", return_value=None); p.start(); self.addCleanup(p.stop)
        self.folder = self.root / "results_catalog"
        self.sources = {"schema_version": 1, "runs": [{"key": "old_run", "display": "Original"}]}
        self.old = {"key": "old_run", "metric": .123, "identity": {"run_id": "unchanged"}}
        self.previous = {"source_spec_sha256": self.fingerprint(self.sources), "runs": [self.old]}
        write_json(self.folder / "sources.json", self.sources)
        write_json(self.folder / "snapshot.json", self.previous)
        for name, content in self.render(self.previous).items():
            (self.folder / name).write_text(content)
        # Archives are independent lossless artifacts; raw transaction must not overwrite them.
        (self.folder / "snapshot.json.gz").write_bytes(b"historical archive sentinel")
        (self.folder / "metrics.csv.gz").write_bytes(b"historical table archive sentinel")

    @staticmethod
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def render(snapshot):
        return {"registry.csv": "key\n" + "\n".join(row["key"] for row in snapshot["runs"]) + "\n"}

    def collect(self, workspace, sources, previous):
        old = {row["key"]: row for row in previous["runs"]}
        return {"source_spec_sha256": self.fingerprint(sources),
                "runs": [deepcopy(old.get(row["key"], row)) for row in sources["runs"]]}

    def catalogue_bytes(self):
        return {p.name:p.read_bytes() for p in self.folder.iterdir() if p.is_file() and not p.name.startswith(".")}

    def test_proposal_registers_only_one_source_valid_winner(self):
        proposed = register.proposal(self.output, root=self.root)
        self.assertEqual(len(proposed["runs"]), 1)
        row = proposed["runs"][0]
        self.assertEqual(row["required_families"], ["coverage"])
        self.assertEqual(row["protocol"], state.PROTOCOL)
        self.assertEqual(row["category"], "cross_combination")
        self.assertTrue(row["valid_selected_primary"])
        self.assertEqual(row["supervision"], "self_supervised")
        self.assertEqual(row["selection"], "artificial/original_valid/selection.json")
        self.assertIn("original greedy matching", row["design_note"])

    def test_atomic_addition_preserves_old_record_and_is_idempotent(self):
        before_archives = {p.name:p.read_bytes() for p in self.folder.glob("*.gz")}
        with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=self.collect):
            result = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(result["new_results"], 1)
            updated = read_json(self.folder / "snapshot.json")
            self.assertEqual(updated["runs"][0], self.old)
            self.assertEqual(len(updated["runs"]), 2)
            second = register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(second, {"status":"already_registered", "new_results":0})
        self.assertEqual({p.name:p.read_bytes() for p in self.folder.glob("*.gz")}, before_archives)

    def test_refuses_collector_mutation_of_prior_result(self):
        before = self.catalogue_bytes()
        def mutate(workspace, sources, previous):
            result = self.collect(workspace, sources, previous)
            result["runs"][0]["metric"] = .9
            return result
        with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=mutate):
            with self.assertRaises(ValueError):
                register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
        self.assertEqual(self.catalogue_bytes(), before)

    def test_refuses_collector_omission_or_unexpected_identity(self):
        for extra in (False, True):
            before = self.catalogue_bytes()
            def mutate(workspace, sources, previous):
                result = self.collect(workspace, sources, previous)
                if extra: result["runs"].append({"key":"unexpected"})
                else: result["runs"].pop()
                return result
            with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=mutate):
                with self.assertRaises(ValueError):
                    register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
            self.assertEqual(self.catalogue_bytes(), before)

    def test_failed_write_rolls_back_raw_catalogue(self):
        before = self.catalogue_bytes()
        original_write = register._atomic_write
        failed = [False]
        def fail_once(path, data):
            if Path(path) == self.folder / "registry.csv" and not failed[0]:
                failed[0] = True
                raise OSError("synthetic interrupted write")
            return original_write(path, data)
        with patch.object(register.catalog, "render", side_effect=self.render), patch.object(register.catalog, "collect", side_effect=self.collect), patch.object(register, "_atomic_write", side_effect=fail_once):
            with self.assertRaises(OSError):
                register.apply_proposal(self.output, workspace=self.workspace, root=self.root)
        self.assertEqual(self.catalogue_bytes(), before)

    def test_inconsistent_prior_source_snapshot_keys_are_rejected(self):
        previous = read_json(self.folder / "snapshot.json")
        previous["runs"][0]["key"] = "different_old_key"
        write_json(self.folder / "snapshot.json", previous)
        for name, content in self.render(previous).items(): (self.folder/name).write_text(content)
        with patch.object(register.catalog, "render", side_effect=self.render):
            with self.assertRaises(ValueError):
                register.apply_proposal(self.output, workspace=self.workspace, root=self.root)


if __name__ == "__main__":
    unittest.main()
