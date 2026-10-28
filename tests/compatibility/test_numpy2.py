"""Exercise both removed-API sites and the wavelet fallback under NumPy 2."""

from pathlib import Path

import numpy as np
import pytest

from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi.data import Data


@pytest.mark.parametrize(
    "significant, step, expected_indices",
    [([], 1, []), ([False] * 5, 2, []), ([True] * 6, 2, [0, 2, 4]),
     ([False, True, True, False, True, True], 2, [1, 4]),
     ([True, False, True], 10, [0])],
)
def test_group_delay_independent_frequencies(significant, step, expected_indices):
    from spectral_connectivity.connectivity import _get_independent_frequencies

    expected = np.zeros(len(significant), dtype=bool)
    expected[expected_indices] = True
    result = _get_independent_frequencies(np.array(significant, dtype=bool), step)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("kind", ["numeric", "strings"])
def test_canonical_coherence_matches_pre_migration_reference(kind):
    # Reference captured with NumPy 1.26.4 and unmodified dependency 1.1.2.
    rng = np.random.default_rng(382)
    shape = (1, 4, 3, 32, 4)
    coefficients = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    conn = sc.Connectivity(
        coefficients, frequencies=np.fft.fftfreq(32), time=np.array([0.]),
    )
    groups = np.array([0, 0, 1, 1] if kind == "numeric" else ["b", "a", "b", "a"])
    actual, labels = conn.canonical_coherence(groups)
    with np.load(Path(__file__).with_name("spectral_reference.npz")) as reference:
        np.testing.assert_allclose(actual, reference[f"canonical_{kind}"], rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(labels, reference[f"labels_{kind}"])


def test_spectral_adapter_does_not_modify_numpy_or_other_modules():
    import spectral_connectivity.connectivity as connectivity
    import spectral_connectivity.transforms as transforms

    assert not hasattr(np, "in1d")
    assert transforms.np is np
    assert connectivity.np.asarray is np.asarray
    # Preserve in1d's flattened result even though isin preserves input shape.
    np.testing.assert_array_equal(
        connectivity.np.in1d([[1, 2], [3, 2]], [2], invert=True),
        [True, False, True, False],
    )


def test_wavelet_three_dimensional_cache_uses_axiserror_fallback():
    from pyspi.statistics.wavelet import CoherenceMagnitude

    data = Data(np.arange(60., dtype=float).reshape(3, 20), normalise=False, detrend=False)
    values = np.arange(27., dtype=float).reshape(3, 3, 3)
    data.mne = {("coh", 1): (values, np.array([0.2, 0.3, 0.4]))}
    actual = CoherenceMagnitude().multivariate(data)
    np.testing.assert_array_equal(
        actual, [[np.nan, 10., 19.], [10., np.nan, 22.], [19., 22., np.nan]],
    )
