"""Algorithm checks for the isolated visual audit (no project training)."""
import numpy as np
from tlinkage import author_cardinality_rejection, linkage, tlinkage_regions


def exhaustive_linkage(p):
    p = p.copy()
    n = len(p)
    active = np.ones(n, dtype=bool)
    groups = [[i] for i in range(n)]
    while True:
        candidates = []
        for i in np.flatnonzero(active):
            for j in np.flatnonzero(active):
                if i >= j:
                    continue
                dot = p[i] @ p[j]
                denominator = p[i] @ p[i] + p[j] @ p[j] - dot
                distance = 1 - dot / denominator if denominator > 0 else 1
                candidates.append((max(0, min(1, distance)), i, j))
        if not candidates:
            break
        distance, i, j = min(candidates)
        if distance >= 1:
            break
        p[i] = np.minimum(p[i], p[j])
        active[j] = False
        groups[i] += groups[j]
    labels = np.zeros(n, dtype=int)
    for k, i in enumerate(np.flatnonzero(active), 1):
        labels[groups[i]] = k
    return labels


def test_global_greedy_matches_exhaustive():
    rng = np.random.default_rng(10)
    for n in [2, 6, 17, 35]:
        p = rng.random((n, 40))
        p[p < .75] = 0
        a, _ = linkage(p)
        b = exhaustive_linkage(p)
        np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)


def test_crossing_lines_are_separated():
    raw = np.ones((52, 52), dtype=np.uint8)
    raw[8:44, 26] = 2
    raw[26, 8:44] = 2
    labels, info = tlinkage_regions(raw)
    assert info['num_regions'] == 2
    assert labels[8, 26] == labels[43, 26] > 0
    assert labels[26, 8] == labels[26, 43] > 0
    assert labels[8, 26] != labels[26, 8]
    assert np.all(labels[raw != 2] == 0)
    assert np.all(labels[raw == 2] > 0)


def test_empty_singleton_and_repeatability():
    raw = np.ones((52, 52), dtype=np.uint8)
    assert not tlinkage_regions(raw)[0].any()
    raw[20, 20] = 2
    assert not tlinkage_regions(raw)[0].any()
    raw[20, 8:44] = 2
    a, _ = tlinkage_regions(raw)
    b, _ = tlinkage_regions(raw)
    np.testing.assert_array_equal(a, b)


def test_author_cardinality_branch_is_explicit():
    labels = np.repeat([1, 2, 3], [2, 10, 50])
    result, info = author_cardinality_rejection(labels)
    assert info['author_at_least_two_models_branch']
    assert (result == 0).sum() == 2
    assert len(np.unique(result[result > 0])) == 2


if __name__ == '__main__':
    for test in [test_global_greedy_matches_exhaustive, test_crossing_lines_are_separated,
                 test_empty_singleton_and_repeatability, test_author_cardinality_branch_is_explicit]:
        test()
        print(test.__name__, 'PASS')
