"""Bind the completed valid-selected regional-J F and six cross references.

Only frozen inputs are loaded. The caller computes the original full-gallery
regional scores before applying cross-combination eligibility.
"""
import os
from pathlib import Path

import numpy as np
import torch

from f_region_area_fusion import inputs as area_inputs
from f_region_area_fusion.pipeline import MATCHING
from f_region_area_fusion.register import proposal as area_proposal
from f_region_area_fusion.run import code_identity as area_code
from f_region_score_search.inputs import _bind
from f_region_structure.state import read_signed
from fgw_cross_test.state import CROSS_SOURCE, cross_pool
from fgj_cross_test.state import VARIANTS as REFERENCES, verify_run as verify_reference
from wbm_retrieval.common import digest, read_json
from .state import VARIANT, SPEC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALID = ROOT / "artificial/f_region_area_fusion_valid_20261120T155825+0800"
DEFAULT_TEST = ROOT / "artificial/f_region_area_fusion_test_20261120T160426+0800"
DEFAULT_REFERENCE = ROOT / "artificial/fgj_cross_test_20261120T193642+0800"


def _path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def _seal(folder, binding, signature, required=()):
    folder = Path(folder).resolve()
    _bind(folder / "completed.json", binding)
    marker = read_json(folder / "completed.json")
    files = marker.get("files")
    if (marker.get("status") != "completed" or marker.get("signature") != signature
            or not isinstance(files, dict) or not set(required).issubset(files)):
        raise ValueError("Frozen source completion is incomplete or incompatible: " + str(folder))
    for name, expected in files.items():
        relative = Path(name)
        path = folder / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(folder) or path.resolve() == folder):
            raise ValueError("Frozen source completion path escapes its directory")
        _bind(path, binding, expected)
    return marker


def _validate_source(data, valid_manifest, test_manifest, input_document, frozen, valid_root):
    """Do not infer the winner from test results or silently accept new inputs."""
    choice = data["frozen_selection"]
    selection = choice["selection"]
    context = test_manifest["context"]
    if (VARIANT != "add_a0p4_b0p25" or SPEC != {"family": "add", "alpha": .4, "beta": .25}
            or selection.get("winner") != VARIANT or selection.get("selected_variants", [None])[0] != VARIANT
            or selection.get("selected_specs", {}).get(VARIANT) != SPEC
            or choice.get("selected_specs", {}).get(VARIANT) != SPEC
            or selection.get("selection_rule", {}).get("split") != "valid"
            or selection.get("selection_rule", {}).get("metric") != "exact/all/Precision@10"
            or selection.get("context_signature") != digest(valid_manifest["context"])
            or frozen != choice):
        raise ValueError("Regional-J source is not the unchanged complete-valid winner")
    if (test_manifest.get("version") != "f_region_area_fusion_v1" or context.get("split") != "test"
            or Path(test_manifest["source_run"]).resolve() != valid_root
            or Path(context["source_run"]).resolve() != valid_root
            or context.get("grid") != choice["selected_specs"]
            or context.get("frozen_selection_signature") != choice["signature"]
            or context.get("source_binding") != data["source_binding"]
            or input_document.get("source_binding") != data["source_binding"]
            or input_document.get("context_signature") != digest(context)
            or input_document.get("source_run") != context.get("source_run")
            or input_document.get("cache_source_run") != str(data["cache_source_run"])
            or context.get("cache_source_run") != str(data["cache_source_run"])
            or context.get("matching") != data["config"]["matching"]
            or any(context.get("matching", {}).get(k) != v for k, v in MATCHING.items())
            or context.get("training") is not False or context.get("test_parameter_search") is not False):
        raise ValueError("Completed regional-J test differs from its frozen input or matcher binding")
    if (context.get("numpy") != np.__version__ or context.get("torch") != torch.__version__
            or any(os.environ.get(k) != "1" for k in
                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))):
        raise ValueError("Use the original NumPy/PyTorch versions and single-thread BLAS environment")
    if context.get("code") != area_code():
        raise ValueError("Frozen regional-J scientific source code changed")


def load(valid_root=DEFAULT_VALID, test_root=DEFAULT_TEST, reference_root=DEFAULT_REFERENCE):
    valid_root, test_root, reference_root = map(_path, (valid_root, test_root, reference_root))
    data = area_inputs.load_test(valid_root)
    valid_manifest = read_signed(valid_root / "run_manifest.json")
    test_manifest = read_signed(test_root / "run_manifest.json")
    context = test_manifest["context"]
    original_inputs = read_json(test_root / "inputs.json")
    frozen = read_signed(test_root / "frozen_selection.json")
    _validate_source(data, valid_manifest, test_manifest, original_inputs, frozen, valid_root)
    checked = area_proposal(test_root)
    if len(checked.get("runs", [])) != 10 or any(r.get("split") != "test" for r in checked["runs"]):
        raise ValueError("Original selected test must have ten completed verified evaluations")
    binding = dict(data["source_binding"])
    _seal(test_root, binding, digest(context), required={"run_manifest.json", "inputs.json",
        "frozen_selection.json", "evaluation_index_test.json", "formula_audit_test.json",
        "status.json", "source_protection_after.json", "source_protection_before.json"})
    for name, expected in context["code"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Scientific code path escapes repository")
        _bind(ROOT / relative, binding, expected)
    winner = test_root / "test" / VARIANT
    winner_signature = digest({"context": digest(context), "split": "test", "variant": VARIANT, "spec": SPEC})
    _seal(winner, binding, winner_signature, required={"evaluation_manifest.json", "run_identity.json",
        "rankings.csv", "metrics.json", "coverage_metrics.json", "per_query.csv", "per_class.csv",
        "coverage_per_query.csv", "coverage_per_class.csv", "verification.json"})
    base = data["protocol"].retrieval.test
    if (len(base), len(base.gallery), base.candidates_per_query) != (7199, 7199, 7198):
        raise ValueError("Expected the complete frozen test pool")
    pool = cross_pool(base, data["bundle"].labels)
    reference, _ = verify_reference(reference_root, check_source=False)
    reference_context = reference["context"]
    if (reference_context.get("derived_protocol") != pool.metadata
            or reference_context.get("base_protocol_signature") != base.protocol_signature
            or reference_context.get("retrieval_protocol_signature") != pool.protocol_signature):
        raise ValueError("Regional-J and six-method references use different candidate eligibility")
    _seal(reference_root, binding, digest(reference_context), required={"run_manifest.json",
        "candidate_protocol.json", "settings.json", "frozen_variants.json", "evaluation_index_test.json",
        "shared_verification.json", "baseline_replay.json", "status.json"})
    references = {}
    for name in REFERENCES:
        folder = reference_root / "test" / name
        ev = read_json(folder / "evaluation_manifest.json")
        _seal(folder, binding, ev["signature"], required={"coverage_metrics.json", "coverage_per_query.csv",
            "coverage_per_class.csv", "rankings.csv", "evaluation_manifest.json", "verification.json"})
        references[name] = str(folder)
    _bind(CROSS_SOURCE, binding, reference_context["source_binding"][str(CROSS_SOURCE.resolve())])
    for value in data["cache"].values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    metadata = {"source_valid": str(valid_root), "source_test": str(test_root),
        "reference_root": str(reference_root), "source_winner_folder": str(winner),
        "reference_folders": references, "source_selection_signature": frozen["selection"]["signature"],
        "source_valid_context_signature": digest(valid_manifest["context"]),
        "source_test_context_signature": digest(context), "source_winner_signature": winner_signature,
        "reference_context_signature": digest(reference_context),
        "cache_source_run": str(data["cache_source_run"]), "cache_folder": str(data["cache_source_run"] / "cache/test"),
        "origin_valid_winner": True, "valid_selection_rank": 1, "source_variant": VARIANT, "source_spec": dict(SPEC),
        "source_scope": "frozen original model and regional cache; complete valid-selected winner; completed original test; completed six-method cross-combination reference",
        "pair_formula_operation_order": "(1-beta)*(alpha*C+(1-alpha)*T)+beta*A",
        "structure_strength": .5, "structure_descriptor": "regional joint_all",
        "matching": "original_F_greedy_sqrt_area", "normalization_scope": "none", "training_required": False,
        "numpy": np.__version__, "torch": torch.__version__, "derived_protocol": pool.metadata}
    binding = dict(sorted(binding.items()))
    return {"cache": data["cache"], "base_pool": base, "pool": pool, "bundle": data["bundle"],
        "matching": data["config"]["matching"], "source_binding": binding, "source_metadata": metadata,
        "source_binding_signature": digest(binding), "frozen_selection": frozen}
