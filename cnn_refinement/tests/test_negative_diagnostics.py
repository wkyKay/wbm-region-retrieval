import numpy as np
import pytest

from cnn_refinement.negative_diagnostics import _expected_exposure, _proxy_counts, audit_negatives
from cnn_refinement.neighbors import build_neighbor_graph


def test_proxy_precision_recall_uses_complete_cross_wafer_denominator():
    graph = {"indptr": np.array([0, 2, 3, 4, 4]), "indices": np.array([1, 2, 0, 0]),
             "token_map_ids": np.array([10, 20, 30, 40])}
    evaluable, hits, relevant = _proxy_counts(graph, np.array([0, 0, 1, -1]))
    np.testing.assert_array_equal(evaluable, [2, 1, 1, 0])
    np.testing.assert_array_equal(hits, [1, 1, 0, 0])
    np.testing.assert_array_equal(relevant, [1, 1, 0, 0])


def test_expected_batch_exclusion_replacement_and_duplicates_are_exact():
    graph = {"indptr": np.array([0, 2, 3, 4, 4]), "indices": np.array([1, 2, 0, 0]),
             "token_map_ids": np.array([10, 20, 30, 40])}
    result = _expected_exposure(graph, [np.array([0, 1]), np.array([2, 3])])
    np.testing.assert_array_equal(result["blocked_outside_batch"], [1, 0, 1, 0])
    np.testing.assert_array_equal(result["expected_replaced_slots"], [2500, 0, 2500, 0])
    np.testing.assert_array_equal(result["probability_zero_replacements"], [0, 1, 0, 1])
    np.testing.assert_allclose(result["expected_duplicate_rate_before"], [1 - 2 / 5000] * 4)
    np.testing.assert_allclose(result["expected_duplicate_rate_after"],
                               [1 - 1 / 5000, 1 - 2 / 5000, 1 - 1 / 5000, 1 - 2 / 5000])


def test_end_to_end_posthoc_audit_is_separate_immutable_and_does_not_change_rng(tmp_path):
    import torch
    teacher = tmp_path / "teacher.npy"
    features = np.ones((12, 512), dtype=np.float64)
    np.save(teacher, features)
    graph = build_neighbor_graph(tmp_path / "neighbors", teacher, teacher, np.arange(12), k=2, device="cpu")
    labels = np.zeros((12, 8), dtype=np.uint8)
    labels[np.arange(12), np.arange(12) % 2] = 1
    inputs = {"refinement_context_signature": "synthetic-context", "protection": {}, "labels": labels,
              "config": {"batch_size": 4}, "datasets": {"train": {"token_map_ids": np.arange(12)}}}
    rng = torch.get_rng_state().clone()
    original_graph_bytes = {p: p.read_bytes() for p in (tmp_path / "neighbors").iterdir()}
    summary = audit_negatives(tmp_path, inputs, graph)
    assert torch.equal(rng, torch.get_rng_state())
    assert summary["tokens"] == 12
    assert summary["observed_diagnostic_sample"]["negative_slots"] == 60000
    assert summary["single_label_proxy"]["recall_micro"] >= 0
    assert (tmp_path / "negative_diagnostics" / "completed.json").is_file()
    assert not (tmp_path / "diagnostics").exists()
    assert original_graph_bytes == {p: p.read_bytes() for p in (tmp_path / "neighbors").iterdir()}
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "negative_diagnostics").iterdir()}
    assert audit_negatives(tmp_path, inputs, graph) == summary
    assert before == {p: p.stat().st_mtime_ns for p in (tmp_path / "negative_diagnostics").iterdir()}
    altered_labels = labels.copy()
    altered_labels[0] = altered_labels[1]
    inputs["labels"] = altered_labels
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_negatives(tmp_path, inputs, graph)
