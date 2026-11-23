"""CPU synthetic interface integration, never a scientific training result."""
import numpy as np
import pytest
import torch

from cnn_refinement import features as feature_module
from cnn_refinement.features import array_identity, extract
from cnn_refinement.neighbors import build_neighbor_graph
from cnn_refinement.training import FilteredNegativeSampler
from f_representation_audit.run import state_hash
from wbm_retrieval.common import file_hash


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("constant", torch.ones(512))
        self.seen = []

    def forward(self, value):
        self.seen.append(value.detach().clone())
        return self.constant[None, :].repeat(len(value), 1) * (1 + value.mean((1, 2, 3)))[:, None]


def prepared_teacher(tmp_path, monkeypatch):
    patch_path = tmp_path / "train_F.npy"
    patches = np.zeros((6, 3, 96, 96), dtype=np.uint8)
    patches[:, :, 10:20, 30:40] = 1
    np.save(patch_path, patches)
    patches = np.load(patch_path, mmap_mode="r")
    checkpoint_path = tmp_path / "best_nce.pt"
    checkpoint_path.write_bytes(b"synthetic checkpoint fixture; no real training")
    model = torch.nn.ModuleDict({"encoder": TinyEncoder()})
    model.eval()
    checkpoint = {"path": str(checkpoint_path), "sha256": file_hash(checkpoint_path),
                  "all_state_sha256": state_hash(model)}
    monkeypatch.setattr(feature_module, "checkpoint_path", lambda *_: checkpoint_path)
    monkeypatch.setattr(feature_module, "load_model", lambda *_: (model, checkpoint))
    inputs = {"refinement_context_signature": "synthetic_train_only_context"}
    ids = np.array([10, 10, 20, 30, 40, 50], dtype=np.int64)
    layout = {"token_map_ids": array_identity(ids)}
    return inputs, model, patches, ids, layout


def test_frozen_feature_files_to_graph_to_exact_5000_negative_slots(tmp_path, monkeypatch):
    inputs, model, patches, ids, layout = prepared_teacher(tmp_path, monkeypatch)
    output = tmp_path / "experiment"
    feature_paths = extract(output, inputs, "baseline", patches, layout, "train",
                            device="cpu", batch_size=3, views=("original", "rot90"))
    assert len(model["encoder"].seen) == 4
    assert torch.equal(model["encoder"].seen[1], torch.rot90(model["encoder"].seen[0], 1, (-2, -1)))
    assert np.load(feature_paths["original"]).dtype == np.float64
    graph = build_neighbor_graph(output / "neighbors", feature_paths["original"], feature_paths["rot90"], ids,
                                 k=2, device="cpu", binding={"labels_used": False})
    assert graph["metadata"]["identity"]["original"]["dtype"] == "<f8"
    assert graph["indices"][graph["indptr"][0]:graph["indptr"][1]].tolist() == [2, 3]
    sampler = FilteredNegativeSampler(graph, 6, "cpu")
    base = torch.arange(1, 6).repeat(1000)
    changes = sampler.replacements(torch.tensor([0]), base, torch.tensor([0]))
    actual = base[None, :].clone()
    actual[changes["rows"], changes["columns"]] = changes["ids"]
    assert actual.shape == (1, 5000)
    assert not torch.isin(actual, torch.tensor([0, 2, 3])).any()
    # Same-wafer token 1 survives: construction's exclusion is not a new train sampler rule.
    assert torch.equal(actual[0][base == 1], base[base == 1])
    snapshots = {p: p.stat().st_mtime_ns for p in (output / "features").rglob("*") if p.is_file()}
    assert extract(output, inputs, "baseline", patches, layout, "train", device="cpu", batch_size=3,
                   views=("original", "rot90")) == feature_paths
    assert snapshots == {p: p.stat().st_mtime_ns for p in (output / "features").rglob("*") if p.is_file()}
    assert len(model["encoder"].seen) == 4


def test_feature_extraction_refuses_changed_order_and_nonbaseline_teacher(tmp_path, monkeypatch):
    inputs, _, patches, _, layout = prepared_teacher(tmp_path, monkeypatch)
    output = tmp_path / "experiment"
    extract(output, inputs, "baseline", patches, layout, "train", device="cpu", views=("original", "rot90"))
    with pytest.raises(ValueError, match="identity mismatch"):
        extract(output, inputs, "baseline", patches, {"different_order": True}, "train",
                device="cpu", views=("original", "rot90"))
    with pytest.raises(ValueError, match="exclusively"):
        extract(output, inputs, "context_dropout", patches, layout, "train",
                device="cpu", views=("original", "rot90"))
