"""Read-only, identity-bound components for the fixed six-formula cross test.

F/G/WBBS are columns of the completed original full-gallery matrices. In
particular G is never recomputed with a shorter BLAS input after eligibility
filtering. J uses the original frozen whole-wafer descriptor and backend.
"""
from pathlib import Path

import numpy as np

from b1_structure_fusion.scoring import StructuralDistances
from fgw_grid_search.cache import FullScoreCache, ids, unit_scores
from fgw_grid_search.pipeline import source_binding as cache_binding
from fgw_topk_test.state import verify_run as verify_fgw
from fgw_cross_test.state import CROSS_SOURCE, cross_pool, load_bundle
from fw_cross_test.state import cross_reference, verify_run as verify_fw
from gj_fusion import inputs as gj_inputs
from gj_fusion.pipeline import Components as OriginalGJComponents
from gj_fusion.verify import audit_components as audit_gj
from wbm_retrieval.common import digest, file_hash, read_json
from fgw_grid_search.state import read_signed

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FGW = ROOT / "artificial/fgw_topk_test_20261119T205138+0800"
DEFAULT_GJ = gj_inputs.DEFAULT_SOURCE
GJ_TEST_ROOT = ROOT / "artificial/gj_fusion_test_20261120T182234+0800"
F_CONTROL_ROOT = ROOT / "artificial/fw_cross_test_20261120T104447+0800"
WBBS_CONTROL_ROOT = ROOT.parent / "compared_methods/artifacts/cross_coverage_seven_test_20261111_v1/wbbs"
KS = [1, 5, 10, 20, 50, 100, 500]


def _bind(path, binding, expected=None):
    return gj_inputs._bind(path, binding, expected)


def _seal(folder, binding, *, signature=None, signature_key="signature", required=()):
    """Bind a whole sealed source, rejecting missing files and escaping paths."""
    folder = Path(folder).resolve()
    _bind(folder / "completed.json", binding)
    done = read_json(folder / "completed.json")
    files = done.get("files")
    if (not isinstance(files, dict) or not set(required).issubset(files)
            or signature is not None and done.get(signature_key) != signature):
        raise ValueError("Incomplete or incompatible source seal: " + str(folder))
    for name, expected in files.items():
        relative = Path(name)
        path = (folder / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or folder not in path.parents:
            raise ValueError("Source completion path escapes its directory")
        _bind(path, binding, expected)
    return done


def _code(values, binding, root=ROOT):
    for name, expected in values.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Scientific code path escapes repository")
        _bind(root / path, binding, expected)


def _gj_test_binding(base_binding, valid_root, binding):
    """Reproduce gj_fusion.run's four additional frozen-valid test bindings."""
    expected = dict(base_binding)
    for name in ("completed.json", "selection.json", "run_manifest.json", "evaluation_index_valid.json"):
        path = (Path(valid_root) / name).resolve()
        sha = _bind(path, binding)
        if str(path) in expected and expected[str(path)] != sha:
            raise ValueError("Frozen G/J valid source conflicts with base source binding")
        expected[str(path)] = sha
    return expected


def _gj_reference(data, binding, device):
    """Tie the underlying B1 G/J sources to the completed new G/J study."""
    root = GJ_TEST_ROOT.resolve()
    manifest = read_signed(root / "run_manifest.json")
    context = manifest["context"]
    signature = digest(context)
    valid_root = Path(manifest["valid_run"]).resolve()
    expected_binding = _gj_test_binding(data["source_binding"], valid_root, binding)
    marker = _seal(root, binding, signature=signature, required={"run_manifest.json", "inputs.json",
        "frozen_selection.json", "evaluation_index_test.json", "baseline_replay.json",
        "component_audit_test.json", "source_protection_after.json", "status.json"})
    inp, status = read_json(root / "inputs.json"), read_json(root / "status.json")
    if (manifest.get("version") != "gj_fusion_v1" or manifest.get("split") != "test"
            or context.get("split") != "test" or context.get("structure_strength") != 2.
            or context.get("device") != device or context.get("angular_component") is not False
            or context.get("local_component") is not False or context.get("test_parameter_search") is not False
            or Path(manifest["source_run"]).resolve() != data["source_run"]
            or inp.get("source_metadata") != data["source_metadata"]
            or inp.get("source_binding") != expected_binding
            or context.get("source_binding") != expected_binding
            or inp.get("context_signature") != signature
            or marker.get("status") != "completed" or status.get("status") != "completed"
            or status.get("context_signature") != signature or status.get("test_evaluated") is not True
            or read_json(root / "source_protection_after.json").get("status") != "unchanged"):
        raise ValueError("Completed G/J study uses different or incomplete frozen G/J sources")
    _code(context["code"], binding)
    frozen = read_signed(root / "frozen_selection.json")
    if frozen["signature"] != context.get("frozen_selection_signature"):
        raise ValueError("Completed G/J test differs from its frozen valid selection")
    valid_manifest = read_signed(valid_root / "run_manifest.json")
    valid_signature = digest(valid_manifest["context"])
    valid_done = _seal(valid_root, binding, signature=valid_signature,
                       required={"run_manifest.json", "selection.json", "evaluation_index_valid.json"})
    if (valid_done.get("status") != "completed" or valid_manifest.get("split") != "valid"
            or valid_manifest.get("source_run") != manifest.get("source_run")
            or read_signed(valid_root / "selection.json") != frozen):
        raise ValueError("Missing original sealed valid selection for completed G/J test")
    index = read_json(root / "evaluation_index_test.json")
    if (index.get("context_signature") != signature or index.get("variants") != context.get("grid")
            or context.get("grid") != frozen.get("selected_specs")):
        raise ValueError("Completed G/J index differs from frozen selection")
    for name in ("G_only", "J_only", "add_a0p5", "product"):
        row = index["results"][name]
        folder = root / "test" / name
        _seal(folder, binding, signature=row["signature"], required={"rankings.csv", "metrics.json",
            "coverage_metrics.json", "evaluation_manifest.json", "verification.json"})
        ev, checked = read_json(folder / "evaluation_manifest.json"), read_json(folder / "verification.json")
        if (file_hash(folder / "completed.json") != row["completed_sha256"]
                or ev.get("structure_strength") != 2. or ev.get("context_signature") != signature
                or ev.get("spec") != context["grid"][name]
                or checked.get("status") != "verified" or checked.get("num_queries") != 7199
                or checked.get("all_queries_full_candidate_topk_exact") is not True):
            raise ValueError("Completed G/J endpoint/fusion evidence changed")
    return {"root": str(root), "signature": manifest["signature"], "context_signature": signature}


def _controls(fgw_root, fgw_manifest, cache, pool, binding):
    """Bind the original F-only and WBBS cross-combination controls for replay."""
    froot = F_CONTROL_ROOT.resolve()
    fm, _ = verify_fw(froot, check_source=False)
    fc = fm["context"]
    if (Path(fc["source_root"]).resolve() != fgw_root or fc["source_signature"] != fgw_manifest["signature"]
            or fc["source_completed_sha256"] != file_hash(fgw_root / "completed.json")
            or fc["provider_source"] != cache_binding(cache) or fc["derived_protocol"] != pool.metadata):
        raise ValueError("Historical F cross control has different full-score source or candidates")
    _seal(froot, binding, signature=fm["signature"], required={"candidate_protocol.json",
        "protocol/cross_combination_candidates.csv", "run_manifest.json", "test/F/completed.json"})
    _code(fc["code"], binding)
    for path, expected in fc.get("external_code", {}).items():
        _bind(path, binding, expected)
    ffolder = froot / "test/F"
    fe = read_json(ffolder / "evaluation_manifest.json")
    _seal(ffolder, binding, signature=fe["signature"], required={"rankings.csv", "coverage_metrics.json",
         "coverage_per_query.csv", "coverage_per_class.csv", "evaluation_manifest.json", "verification.json"})
    if fe.get("spec") != {"family": "local_f", "formula": "F"}:
        raise ValueError("Historical F control is not the optimal-region F endpoint")
    reference = cross_reference(pool.metadata)
    for name, expected in reference["files"].items():
        _bind(Path(reference["root"]) / name, binding, expected)
    wroot = WBBS_CONTROL_ROOT.resolve()
    _bind(wroot / "evaluation_manifest.json", binding)
    wm = read_signed(wroot / "evaluation_manifest.json")
    done = _seal(wroot, binding, signature=wm["signature"], signature_key="evaluation_signature",
        required={"test/rankings.csv", "test/coverage_metrics.json", "test/coverage_per_query.csv",
                  "test/coverage_per_class.csv", "test/prepared/features.npz"})
    if (wm.get("method") != "wbbs" or wm.get("split") != "test" or wm.get("ks") != KS
            or wm.get("tie_seed") != 2026 or wm.get("derived_protocol") != pool.metadata
            or wm.get("query_count") != 7199 or wm.get("gallery_count") != 7199
            or wm.get("metric_version") != "coverage_v1" or wm.get("smoke") is not False
            or done.get("num_queries") != 7199 or done.get("test_evaluated") is not True):
        raise ValueError("Historical WBBS control differs from the complete cross protocol")
    _code(wm["code"], binding, ROOT.parent / "compared_methods")
    for name, expected in wm["sources"]["retrieval_protocol"].items():
        _bind(ROOT.parent / "compared_methods/protocol/retrieval_v1" / name, binding, expected)
    _bind(ROOT.parent / "compared_methods/protocol/deduplicated/split.csv", binding,
          wm["sources"]["split_file"])
    for folder in (ffolder, wroot / "test"):
        report = read_json(folder / "coverage_metrics.json")
        if (report.get("metadata", {}).get("derived_protocol") != pool.metadata
                or report.get("metric_version") != "coverage_v1" or report.get("num_queries") != 7199
                or report.get("ks") != KS or report.get("split") != "test"):
            raise ValueError("Historical cross control report has different coverage semantics")
    return {"F_ONLY": str(ffolder), "WBBS": str(wroot / "test")}


def load_inputs(fgw_root=DEFAULT_FGW, gj_root=DEFAULT_GJ, device="cuda"):
    """Load immutable full matrices and G/J inputs; never run encoders or F scoring."""
    fgw_root = (ROOT / Path(fgw_root).expanduser()).resolve()
    gj = gj_inputs.load_inputs(gj_root, split="test", device=device)
    binding = dict(gj["source_binding"])
    gj_reference = _gj_reference(gj, binding, device)
    manifest, _ = verify_fgw(fgw_root, check_cache=False)
    _seal(fgw_root, binding, signature=manifest["signature"], required={"run_manifest.json",
        "inputs.json", "source_audit.json", "component_cache/test/plan.json",
        "component_cache/test/completed/completed.json"})
    context, provenance = manifest["context"], manifest["context"]["provenance"]
    for key in ("code", "grid_code"):
        _code(context[key], binding)
    _code(provenance["source_code"], binding)
    if (provenance["fg"]["global_features"] != gj["source_metadata"]["global_features"]
            or provenance["fg"]["descriptor_completed_sha256"] !=
            binding[str(Path(gj["source_metadata"]["descriptor_root"]) / "completed.json")]):
        raise ValueError("F/G/W matrices and G/J study use different frozen components")
    cache = FullScoreCache(fgw_root / "component_cache/test")
    inp = read_signed(fgw_root / "inputs.json")
    if (cache.signature != digest({"inputs": inp["signature"], "operation": "full_F_G_W_scores"})
            or cache.protocol_signature != gj["pool"].protocol_signature
            or not np.array_equal(cache.query_ids, list(gj["pool"]))
            or not np.array_equal(cache.gallery_ids, gj["pool"].gallery)):
        raise ValueError("Full-score cache and frozen G/J protocol identities differ")
    # FullScoreCache has checked every matrix's hash, dtype, finite range and self NaN.
    for name, expected in cache.completion["files"].items():
        binding[str((cache.marker.parent / name).resolve())] = expected
    bundle, base, labels_origin = load_bundle(fgw_root, cache)
    if (not np.array_equal(bundle.labels[base.gallery], gj["bundle"].labels[base.gallery])
            or not np.array_equal(bundle.class_ids[base.gallery], gj["bundle"].class_ids[base.gallery])):
        raise ValueError("Original F/G/W and G/J labels or class mapping differ")
    _bind(labels_origin["source_label_report"], binding, labels_origin["source_label_report_sha256"])
    for name, expected in labels_origin["retrieval_files"].items():
        _bind(ROOT / "protocol/retrieval_v1" / name, binding, expected)
    _bind(ROOT / "protocol/deduplicated/split.csv", binding, labels_origin["split_sha256"])
    pool = cross_pool(base, bundle.labels)
    _bind(CROSS_SOURCE, binding)
    controls = _controls(fgw_root, manifest, cache, pool, binding)
    metadata = {"split": "test", "fgw_root": str(fgw_root), "gj_root": str(gj["source_run"]),
        "fgw_signature": manifest["signature"], "full_cache": cache_binding(cache),
        "gj_source": gj["source_metadata"], "gj_completed_reference": gj_reference,
        "controls": controls, "labels_origin": labels_origin, "derived_protocol": pool.metadata,
        "base_protocol_signature": base.protocol_signature, "protocol_signature": pool.protocol_signature,
        "f_definition": "cc8 optimal one-to-one assignment; original CNN*cosine radial/relative-area score",
        "g_definition": "original full-except-self float32 cosine clipped to [0,1], cached as float64",
        "j_definition": "exp(-2*D_H) of original whole-wafer joint128 descriptor", "j_strength": 2.,
        "device": device, "training_required": False, "regional_scoring_recomputed": False,
        "global_scoring_recomputed_for_filtering": False}
    binding = dict(sorted(binding.items()))
    return {"bundle": bundle, "base_pool": base, "pool": pool, "cache": cache,
        "protocol": gj["protocol"], "global_ids": gj["global_ids"], "global_features": gj["global_features"],
        "descriptors": gj["descriptors"], "source_binding": binding, "source_binding_signature": digest(binding),
        "source_metadata": metadata}


class Components:
    """Subset immutable full scores and compute J only for eligible candidates."""
    def __init__(self, data, device="cuda"):
        self.cache, self.pool, self.device = data["cache"], data["pool"], device
        d = data["descriptors"]
        self.distances = StructuralDistances(d["sample_ids"], d["joint128"], d["empty"], device)
        if (not np.array_equal(self.cache.query_ids, list(self.pool))
                or not np.array_equal(self.cache.gallery_ids, self.pool.gallery)):
            raise ValueError("Component cache IDs differ from eligible query/gallery identities")

    def __call__(self, query, candidates):
        if (isinstance(query, (bool, np.bool_)) or not isinstance(query, (int, np.integer))
                or int(query) not in self.cache.rows):
            raise ValueError("Expected a frozen original query ID")
        query, candidates = int(query), ids(candidates)
        eligible = self.pool[query]
        if not np.isin(candidates, eligible, assume_unique=True).all():
            raise ValueError("Component request contains ineligible candidates")
        columns = [self.cache.columns[int(x)] for x in candidates]
        result = {name: unit_scores(a[self.cache.rows[query], columns], candidates.shape, name,
                                   strict_dtype=True)
                  for name, a in self.cache.matrices.items()}
        result["joint_distance"] = unit_scores(self.distances.for_query(query, candidates)["J"],
                                                candidates.shape, "joint_distance", strict_dtype=True)
        return result


def audit_components(data, provider):
    """Check 740 independent scalar pairs and original full-row G arithmetic."""
    def source_pair(query, candidates):
        values = provider(query, candidates)
        return values["G"], values["joint_distance"]
    report = audit_gj(data, source_pair, max_candidates=20)
    if report["num_queries"] != 37 or report["candidate_pairs"] != 740:
        raise ValueError("Component audit requires 37 complete-label representatives and 740 pairs")
    original = OriginalGJComponents(data, device=provider.device)
    checked = 0
    cache = data["cache"]
    for query in report["query_ids"]:
        candidates = np.asarray(data["base_pool"][query])
        expected, _ = original(query, candidates)
        columns = [cache.columns[int(x)] for x in candidates]
        actual = np.asarray(cache.matrices["G"][cache.rows[query], columns])
        if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
            error = float(np.max(np.abs(actual - expected)))
            raise ValueError(f"Cached full-gallery G is not bitwise identical: query={query}, max_error={error}")
        checked += len(candidates)
    report.update(full_global_candidates_checked=checked, max_full_global_cache_error=0.,
                  full_global_cache_bitwise_identical=True,
                  full_global_scope="Original gj_fusion.Components over full gallery except self, before filtering")
    return report
