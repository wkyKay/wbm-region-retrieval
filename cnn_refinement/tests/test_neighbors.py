"""Independent small-data oracle for frozen false-negative graph construction."""
import numpy as np
import pytest

from cnn_refinement.neighbors import _cpu_topk, _cuda_topk, build_neighbor_graph, load_graph


def features(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros((len(values), 512), dtype=np.float32)
    result[:, :values.shape[1]] = values
    return result


def oracle(original, rotated, ids, k):
    views = []
    for raw in (original, rotated):
        x = raw.astype(np.float64)
        x = (x / np.sqrt((x * x).sum(axis=1))[:, None]).astype(np.float32)
        scores = x @ x.T
        neighbors = []
        for query in range(len(x)):
            eligible = [i for i in range(len(x)) if ids[i] != ids[query]]
            ordered = sorted(eligible, key=lambda i: (-float(scores[query, i]), i))
            neighbors.append(set(ordered[:k]))
        views.append(neighbors)
    return [{i for i in views[0][q] & views[1][q] if q in views[0][i] and q in views[1][i]}
            for q in range(len(ids))]


def rows(graph):
    return [set(graph["indices"][graph["indptr"][i]:graph["indptr"][i + 1]].tolist())
            for i in range(len(graph["token_map_ids"]))]


def test_exhaustive_two_view_mutual_oracle_and_input_preservation(tmp_path):
    generator = np.random.default_rng(121)
    original = features(generator.normal(size=(21, 9)))
    rotated = features(generator.normal(size=(21, 9)))
    ids = np.repeat(np.arange(7), 3).astype(np.int64)
    copies = [a.copy() for a in (original, rotated, ids)]
    graph = build_neighbor_graph(tmp_path / "graph", original, rotated, ids, k=6, device="cpu", block_size=4)
    assert rows(graph) == oracle(original, rotated, ids, 6)
    assert graph["indptr"].dtype == np.int64
    assert graph["indices"].dtype == np.int32
    assert not graph["indices"].flags.writeable
    for before, after in zip(copies, (original, rotated, ids)):
        np.testing.assert_array_equal(before, after)
    for q, candidates in enumerate(rows(graph)):
        assert len(candidates) <= 6
        assert all(ids[q] != ids[c] and q in rows(graph)[c] for c in candidates)


def test_exact_ties_choose_token_order_before_mutual_filter(tmp_path):
    x = features([[1, 0]] * 7)
    ids = np.array([3, 3, 7, 9, 10, 11, 12], dtype=np.int64)
    graph = build_neighbor_graph(tmp_path, x, x, ids, k=2, device="cpu", block_size=2)
    top = np.load(tmp_path / "topk_original.npy")
    np.testing.assert_array_equal(top[0], [2, 3])
    np.testing.assert_array_equal(top[1], [2, 3])
    np.testing.assert_array_equal(top[2], [0, 1])
    assert rows(graph) == [{2, 3}, {2, 3}, {0, 1}, {0, 1}, set(), set(), set()]
    assert graph["statistics"]["degree_zero_count"] == 3
    assert graph["statistics"]["degree_max"] == 2


@pytest.mark.parametrize("ids, expected", [([0, 0, 0], [set(), set(), set()]),
                                         ([0, 0, 1], [{2}, {2}, {0, 1}])])
def test_fewer_than_k_eligible_or_no_cross_wafer_neighbors(tmp_path, ids, expected):
    x = features([[1, 0], [0, 1], [1, 1]])
    graph = build_neighbor_graph(tmp_path, x, x, np.array(ids), k=500, device="cpu")
    assert rows(graph) == expected
    assert int(np.diff(graph["indptr"]).max()) <= 500


def test_both_views_and_bidirectionality_are_required(tmp_path):
    original = features([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, 0]])
    rotated = original[[0, 2, 1, 3, 4]]
    ids = np.arange(5)
    graph = build_neighbor_graph(tmp_path, original, rotated, ids, k=1, device="cpu")
    assert rows(graph) == oracle(original, rotated, ids, 1)
    assert all(not row for row in rows(graph))


def test_complete_reuse_is_read_only_and_changed_identity_is_rejected(tmp_path):
    x = features([[1, 0], [0, 1], [1, 1]])
    ids = np.arange(3)
    first = build_neighbor_graph(tmp_path, x, x, ids, k=2, device="cpu", binding={"teacher_sha256": "frozen"})
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}
    second = build_neighbor_graph(tmp_path, x, x, ids, k=2, device="cpu", binding={"teacher_sha256": "frozen"})
    assert first["signature"] == second["signature"]
    assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="signature changed"):
        build_neighbor_graph(tmp_path, x, x, ids, k=1, device="cpu", binding={"teacher_sha256": "frozen"})
    with pytest.raises(ValueError, match="signature changed"):
        build_neighbor_graph(tmp_path, x, x, ids, k=2, device="cpu", binding={"teacher_sha256": "other"})


def test_feature_file_provenance_and_tamper_detection(tmp_path):
    x = features([[1, 0], [0, 1], [1, 1]])
    source = tmp_path / "teacher.npy"
    np.save(source, x)
    folder = tmp_path / "graph"
    graph = build_neighbor_graph(folder, source, source, np.arange(3), k=2, device="cpu")
    binding = graph["metadata"]["identity"]["original"]
    assert binding["source_path"] == str(source.resolve())
    assert len(binding["sha256"]) == len(binding["file_sha256"]) == 64
    (folder / "indices.npy").write_bytes((folder / "indices.npy").read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        load_graph(folder)


def test_incomplete_output_is_not_overwritten(tmp_path):
    marker = tmp_path / "metadata.json"
    marker.write_text("preserve interrupted attempt")
    x = features([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="Incomplete graph"):
        build_neighbor_graph(tmp_path, x, x, np.arange(2), device="cpu")
    assert marker.read_text() == "preserve interrupted attempt"


@pytest.mark.parametrize("change", ["zero_norm", "nonfinite", "shape", "not_h512", "map_float", "negative_id", "zero_k", "bool_k", "zero_block"])
def test_invalid_inputs_fail_before_output_creation(tmp_path, change):
    x = features([[1, 0], [0, 1]])
    y = x.copy()
    ids = np.arange(2)
    kwargs = {"k": 1, "device": "cpu", "block_size": 1}
    if change == "zero_norm":
        x[0] = 0
    elif change == "nonfinite":
        x[0, 0] = np.nan
    elif change == "shape":
        y = y[:1]
    elif change == "not_h512":
        x, y = x[:, :2], y[:, :2]
    elif change == "map_float":
        ids = ids.astype(np.float32)
    elif change == "negative_id":
        ids[0] = -1
    elif change == "zero_k":
        kwargs["k"] = 0
    elif change == "bool_k":
        kwargs["k"] = True
    elif change == "zero_block":
        kwargs["block_size"] = 0
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        build_neighbor_graph(output, x, y, ids, **kwargs)
    assert not output.exists()


def test_cuda_exact_ties_match_cpu_if_available(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; exhaustive CPU oracle still runs")
    x = features([[1, 0]] * 8)
    ids = np.repeat(np.arange(4), 2)
    cpu = build_neighbor_graph(tmp_path / "cpu", x, x, ids, k=3, device="cpu", block_size=3)
    gpu = build_neighbor_graph(tmp_path / "gpu", x, x, ids, k=3, device="cuda:0", block_size=3)
    assert rows(cpu) == rows(gpu)
    np.testing.assert_array_equal(np.load(tmp_path / "cpu" / "topk_original.npy"),
                                  np.load(tmp_path / "gpu" / "topk_original.npy"))


@pytest.mark.parametrize("width", [1, 3, 5])
def test_torch_threshold_tie_repair_and_insufficient_candidates_on_cpu(width):
    # Exercise the exact CUDA-selection tensor code even on a CPU-only developer host.
    torch = pytest.importorskip("torch")
    scores = np.array([[1, 1, 1, 0, -np.inf], [-np.inf] * 5,
                       [-np.inf, -np.inf, 0.5, 0.5, -np.inf],
                       [0.1, 0.3, 0.2, 0.3, 0.3]], dtype=np.float32)
    np.testing.assert_array_equal(_cuda_topk(torch.from_numpy(scores), width), _cpu_topk(scores, width))
