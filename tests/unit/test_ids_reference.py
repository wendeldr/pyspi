"""Independent small references for IDS feature correlations and reductions.

This locks the current feature/p-value construction, not every statistical
interpretation or parameter choice of the estimator.
"""

import math

import numpy as np
import pytest

from pyspi.lib.ids.numpy_dependence import compute_IDS_numpy


X = np.array([[-1.1, 0.2], [0.5, -0.7], [1.4, 0.8], [-0.2, 1.9], [0.9, -1.3], [-0.8, 0.6]])
Y = np.array([[0.4, -1.2, 0.1], [-0.9, 0.6, 1.2], [1.0, 1.4, -0.7],
              [1.7, -0.3, 0.9], [-0.4, 0.8, -1.1], [0.2, 1.1, 0.5]])


def reference(x, y, norm, bandwidth):
    """Two explicit feature terms; pairwise Pearson products, then reduction."""
    out = np.empty((x.shape[1], y.shape[1]))
    for i in range(x.shape[1]):
        for j in range(y.shape[1]):
            correlations = []
            for term_x in (0, 1):
                for term_y in (0, 1):
                    fx = [math.exp(-bandwidth * v * v) * v ** term_x for v in x[:, i]]
                    fy = [math.exp(-bandwidth * v * v) * v ** term_y for v in y[:, j]]
                    fx = [v - sum(fx) / len(fx) for v in fx]
                    fy = [v - sum(fy) / len(fy) for v in fy]
                    numerator = sum(a * b for a, b in zip(fx, fy))
                    denominator = math.sqrt(sum(a * a for a in fx) * sum(b * b for b in fy))
                    correlations.append(abs(numerator / denominator) if denominator else 0.0)
            if norm == "max":
                out[i, j] = max(correlations)
            elif norm == 1:
                out[i, j] = sum(correlations) / 4
            else:
                out[i, j] = math.sqrt(sum(c * c for c in correlations) / 4)
    return out


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("norm", ["max", 1, 2])
@pytest.mark.parametrize("bandwidth", [0.25, 0.5])
def test_ids_matches_pairwise_feature_reference(cross, norm, bandwidth):
    y = Y if cross else X
    actual = compute_IDS_numpy(X, Y=y if cross else None, num_terms=2,
                               p_norm=norm, bandwidth_term=bandwidth)
    expected = reference(X, y, norm, bandwidth)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cross", [False, True])
def test_ids_pvalues_follow_explicit_permutations(monkeypatch, cross):
    # Noise is constructed so argsort yields these independently prescribed
    # row permutations. The oracle shuffles columns directly, without argsort.
    permutations = [np.array([2, 5, 0, 4, 1, 3]), np.array([5, 3, 1, 0, 4, 2]),
                    np.array([1, 4, 3, 2, 5, 0])]
    noises, pairs = [], []
    for trial in range(3):
        permuted = []
        for source in ([X, Y] if cross else [X]):
            noise = np.empty(source.shape)
            shuffled = np.empty_like(source)
            for col in range(source.shape[1]):
                perm = permutations[(trial + col) % 3]
                noise[perm, col] = np.arange(6)
                shuffled[:, col] = source[perm, col]
            noises.append(noise)
            permuted.append(shuffled)
        pairs.append((permuted[0], permuted[1] if cross else permuted[0]))
    queue = iter(noises)
    def normal(*, size):
        noise = next(queue)
        assert noise.shape == size
        return noise
    monkeypatch.setattr(np.random, "normal", normal)
    observed = reference(X, Y if cross else X, "max", 0.5)
    count = sum(reference(x, y, "max", 0.5) > observed for x, y in pairs)
    actual, pvalues = compute_IDS_numpy(X, Y=Y if cross else None, num_terms=2,
                                       p_val=True, num_tests=3)
    np.testing.assert_allclose(actual, observed, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(pvalues, count / 3)
    assert list(queue) == []
