"""Independent phase-lag definitions, debiasing preservation and shared work."""

from unittest.mock import patch

import numpy as np
import pytest

from pyspi._cache import ArrayCache
from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi.base import Directed, Signed, Undirected, Unsigned
from pyspi.data import Data
from pyspi.statistics import spectral


def connection(imaginary):
    imaginary = np.asarray(imaginary, float)
    n, frequencies = imaginary.shape
    coefficients = np.ones((1, 1, n, 2 * frequencies, 2), complex)
    coefficients[0, 0, :, 1:frequencies+1, 1] -= 1j * imaginary
    return SpectralConnectivity(coefficients, np.fft.fftfreq(2 * frequencies), np.array([0.]))


@pytest.mark.parametrize("ordinary,signed", [
    (spectral.PhaseLagIndex, spectral.SignedPhaseLagIndex),
    (spectral.WeightedPhaseLagIndex, spectral.SignedWeightedPhaseLagIndex),
])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_magnitude_is_taken_before_band_summary(ordinary, signed, statistic):
    # Per-frequency signed values +.6 and -.6. abs(mean) would incorrectly be 0.
    conn = connection([[1, -1], [1, -1], [1, -1], [1, -1], [-1, 1]])
    data = Data(np.ones((2, 8)), normalise=False)
    kwargs = dict(fmin=.25, fmax=.5, statistic=statistic)
    with patch("pyspi.statistics.spectral.spectral_workspace", return_value=conn):
        strength = ordinary(**kwargs).multivariate(data)
        sign = signed(**kwargs).multivariate(data)
    np.testing.assert_allclose(strength[0, 1], .6)
    np.testing.assert_allclose(strength, strength.T)
    np.testing.assert_allclose(sign[0, 1], 0 if statistic == "mean" else .6)
    assert isinstance(ordinary(), Unsigned) and isinstance(ordinary(), Undirected)
    assert isinstance(signed(), Signed) and not isinstance(signed(), Undirected)
    assert isinstance(signed(), Directed)
    assert ordinary().identifier != signed().identifier


def test_weighted_definition_and_debiased_pair_products():
    imaginary = np.array([[10, 1], [10, 1], [-1, 1], [-1, -1], [-1, -1.]])
    conn = connection(imaginary)
    n = len(imaginary)
    actual = {name: conn.measure(name)[0, 1:, 0, 1] for name in (
        "phase_lag_index", "signed_phase_lag_index", "weighted_phase_lag_index",
        "signed_weighted_phase_lag_index", "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index", "phase_locking_value",
        "pairwise_phase_consistency")}
    for f in range(imaginary.shape[1]):
        y = imaginary[:, f]
        signs = np.sign(y)
        unit = (1+1j*y) / np.abs(1+1j*y)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        expected = {
            "signed_phase_lag_index": signs.mean(),
            "phase_lag_index": abs(signs.mean()),
            "signed_weighted_phase_lag_index": y.sum()/abs(y).sum(),
            "weighted_phase_lag_index": abs(y.sum())/abs(y).sum(),
            "debiased_squared_phase_lag_index": np.mean([signs[i]*signs[j] for i, j in pairs]),
            "debiased_squared_weighted_phase_lag_index": (
                sum(y[i]*y[j] for i, j in pairs)/sum(abs(y[i]*y[j]) for i, j in pairs)),
            "phase_locking_value": abs(unit.mean()),
            "pairwise_phase_consistency": np.mean([(unit[i]*unit[j].conjugate()).real
                                                    for i, j in pairs]),
        }
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name][f], value, atol=1e-14)
    assert actual["debiased_squared_phase_lag_index"][1] == pytest.approx(-.2)
    assert actual["debiased_squared_weighted_phase_lag_index"][1] == pytest.approx(-.2)


@pytest.mark.parametrize("name", ["debiased_squared_phase_lag_index",
                                 "debiased_squared_weighted_phase_lag_index",
                                 "pairwise_phase_consistency"])
def test_negative_debiased_estimates_are_preserved_and_symmetric(name):
    # Alternating quadrature phases: correction produces a finite negative estimate.
    coeff = np.ones((1, 1, 4, 8, 2), complex)
    coeff[0, 0, :, :, 1] = np.array([1j, -1j, 1j, -1j])[:, None]
    conn = SpectralConnectivity(coeff, np.fft.fftfreq(8), np.array([0.]))
    value = conn.measure(name)
    np.testing.assert_allclose(value[0, :, 0, 1], -1/3)
    np.testing.assert_allclose(value[0, :, 1, 0], -1/3)


@pytest.mark.parametrize("cls", [spectral.DebiasedSquaredPhaseLagIndex,
                                 spectral.DebiasedSquaredWeightedPhaseLagIndex,
                                 spectral.PairwisePhaseConsistency])
def test_debiased_metadata_describes_negative_values_without_direction(cls):
    spi = cls()
    assert spi.issigned() and "signed" in spi.labels and "unsigned" not in spi.labels
    assert isinstance(spi, Undirected) and "undirected" in spi.labels


def test_weighted_moments_share_cross_products_and_reuse_cached_moments():
    conn = spectral_workspace(Data(np.random.default_rng(42).normal(size=(4, 64))), 1)
    with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduce:
        conn.measure("signed_weighted_phase_lag_index")
        conn.measure("weighted_phase_lag_index")
        conn.measure("debiased_squared_weighted_phase_lag_index")
        assert [call.args[0] for call in reduce.call_args_list] == [
            ["imag", "abs_imag"], ["imag_pairs", "abs_imag_pairs"]]


def test_multimoment_cross_products_are_bounded_and_formed_only_once(monkeypatch):
    conn = spectral_workspace(Data(np.random.default_rng(41).normal(size=(5, 64))), 1)
    monkeypatch.setattr(conn, "block_bytes", 5*5*16*2)
    multiply = np.matmul
    sizes = []
    def record(a, b, **kwargs):
        value = multiply(a, b, **kwargs)
        sizes.append(value.nbytes)
        return value
    with patch("pyspi._spectral.np.matmul", side_effect=record):
        conn.measure("debiased_squared_weighted_phase_lag_index")
    joint_sizes = sizes.copy()
    sizes.clear()
    with patch("pyspi._spectral.np.matmul", side_effect=record):
        conn._reduce_phase(["sign"])
    assert joint_sizes == sizes and max(sizes) <= conn.block_bytes


def test_batched_cache_preserves_requested_hits_during_eviction():
    cache = ArrayCache(max_bytes=32)
    first = cache.get("first", lambda: np.arange(4.))
    values = cache.get_many(["first", "second", "third"],
                            lambda missing: {key: np.ones(4) for key in missing})
    assert values["first"] is first
    assert cache.nbytes == 32 and list(cache.entries) == ["third"]
    assert len(values) == 3


PHASE_CLASSES = [spectral.PhaseLockingValue, spectral.PhaseLagIndex, spectral.SignedPhaseLagIndex,
                 spectral.WeightedPhaseLagIndex, spectral.SignedWeightedPhaseLagIndex,
                 spectral.DebiasedSquaredPhaseLagIndex, spectral.DebiasedSquaredWeightedPhaseLagIndex,
                 spectral.PairwisePhaseConsistency]


def test_registered_phase_moments_form_in_one_pass_and_match_separate_computation(tmp_path):
    import yaml
    from pyspi._spectral import PHASE_MOMENTS, PHASE_MOMENT_ORDER
    from pyspi.calculator import Calculator
    raw = np.random.default_rng(13).normal(size=(6, 96))
    bands = [{"fmin": 0, "fmax": 0.5}, {"fmin": 0.1, "fmax": 0.3, "statistic": "max"}]
    # Each class on its own Data: nothing registered, so only its own moments form.
    expected = {}
    for cls in PHASE_CLASSES:
        for band in bands:
            spi = cls(**band)
            fresh = Data(raw)
            expected[spi.identifier] = spi.multivariate(fresh)
            formed = {key[1] for key in spectral_workspace(fresh, 1)._arrays.entries if key[0] == "phase_statistic"}
            assert formed <= set(PHASE_MOMENTS[spi.measure]) | {"nonzero"}
    config = {".statistics.spectral": {cls.__name__: {"configs": bands} for cls in reversed(PHASE_CLASSES)}}
    path = tmp_path / "phase.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    calc = Calculator(dataset=Data(raw), configfile=str(path))
    conn = spectral_workspace(calc.dataset, 1)
    default_budget = conn._arrays.max_bytes
    with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduce:
        calc.compute()
    assert reduce.call_count == 1
    assert reduce.call_args.args[0] == list(PHASE_MOMENT_ORDER)
    assert set(calc.spis) == set(expected)
    for identifier, value in expected.items():
        np.testing.assert_array_equal(calc.table[identifier].to_numpy(), value, err_msg=identifier)
    assert conn.phase_requests == set(PHASE_MOMENT_ORDER)
    retained = sum(ArrayCache.size(v) for k, v in conn._arrays.entries.items() if k[0] in ("phase_statistic", "measure"))
    assert conn._arrays.max_bytes == default_budget >= retained


def test_phase_registration_is_scoped_to_its_sampling_frequency_and_keeps_the_default_budget_otherwise():
    data = Data(np.random.default_rng(14).normal(size=(3, 64)))
    spectral.PhaseLockingValue(fs=2)._register_shared_work(data)
    spectral.CoherenceMagnitude()._register_shared_work(data)
    assert spectral_workspace(data, 1).phase_requests == set()
    assert spectral_workspace(data, 1)._arrays.max_bytes == ArrayCache().max_bytes
    assert spectral_workspace(data, 2).phase_requests == {"unit"}
    assert spectral_workspace(data, 2)._arrays.max_bytes == ArrayCache().max_bytes
    conn = spectral_workspace(data, 1)
    with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduce:
        conn.measure("signed_phase_lag_index")
    assert reduce.call_args.args[0] == ["sign", "nonzero"]


def test_configured_cache_limit_changes_work_sharing_without_changing_outputs(tmp_path):
    import yaml
    from pyspi.calculator import Calculator
    raw = np.random.default_rng(15).normal(size=(6, 96))
    path = tmp_path / "phase.yaml"
    path.write_text(yaml.safe_dump({".statistics.spectral": {
        cls.__name__: {"configs": [{}]} for cls in PHASE_CLASSES}}))
    results, passes = [], []
    for mib in (.03, 1):
        calc = Calculator(dataset=Data(raw), configfile=str(path), spectral_cache_mib=mib)
        calc.dataset._spectral_cache_bytes = calc._spectral_cache_bytes
        conn = spectral_workspace(calc.dataset, 1)
        with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduced:
            calc.compute()
        passes.append(reduced.call_count)
        results.append(calc.table.to_numpy().copy())
        assert conn._arrays.max_bytes == int(mib * 2**20)
        assert conn._arrays.nbytes <= conn._arrays.max_bytes
    assert passes[0] > passes[1] == 1
    np.testing.assert_array_equal(*results)


def test_signed_unsigned_and_debiased_variants_share_signs_and_are_order_independent():
    raw = np.random.default_rng(12).normal(size=(3, 64))
    classes = [spectral.PhaseLagIndex, spectral.SignedPhaseLagIndex,
               spectral.DebiasedSquaredPhaseLagIndex]
    data = Data(raw)
    conn = spectral_workspace(data, 1)
    with patch.object(conn, "_reduce_phase", wraps=conn._reduce_phase) as reduce:
        actual = [cls().multivariate(data) for cls in classes]
        assert reduce.call_count == 1
    fresh = Data(raw)
    for cls, expected in reversed(list(zip(classes, actual))):
        np.testing.assert_allclose(cls().multivariate(fresh), expected)


@pytest.mark.parametrize("imaginary", [
    [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, -1, 0, 0],
    [1, -1, 1, -1, 0], [1, 1, 1, -1, -1],
])
def test_zero_signs_follow_distinct_observation_pair_definition(imaginary):
    conn = connection(np.asarray(imaginary)[:, None])
    signs = np.sign(imaginary)
    expected = np.mean([a*b for i, a in enumerate(signs)
                        for j, b in enumerate(signs) if i != j])
    result = conn.measure("debiased_squared_phase_lag_index")
    np.testing.assert_allclose(result[0, 1, 0, 1], expected, atol=1e-15)
    np.testing.assert_allclose(result[0, 1, 1, 0], expected, atol=1e-15)


@pytest.mark.parametrize("scale", [1e-100, 1e-40, 1e-10, 1, 1e10, 1e40, 1e100])
@pytest.mark.parametrize("imaginary", [[1, 2, 3, 4, 5], [2, -3, 4, -1, 0],
                                      [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
def test_weighted_ratios_are_scale_invariant_and_require_usable_weights(scale, imaginary):
    y = np.asarray(imaginary, float)
    conn = connection(y[:, None])
    conn.fourier_coefficients *= scale
    weighted = y.sum() / abs(y).sum() if np.any(y) else np.nan
    pairs = [(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j]
    pair_weight = sum(abs(a*b) for a, b in pairs)
    debiased = sum(a*b for a, b in pairs) / pair_weight if pair_weight else np.nan
    for name, expected in [("signed_weighted_phase_lag_index", weighted),
                           ("weighted_phase_lag_index", abs(weighted)),
                           ("debiased_squared_weighted_phase_lag_index", debiased)]:
        result = conn.measure(name)
        np.testing.assert_allclose(result[0, 1, 0, 1], expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("cls", [spectral.WeightedPhaseLagIndex,
                                 spectral.SignedWeightedPhaseLagIndex,
                                 spectral.DebiasedSquaredWeightedPhaseLagIndex])
@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_undefined_weighted_bins_are_skipped_and_all_undefined_band_stays_nan(cls, statistic):
    conn = connection([[0, 1], [0, 1], [0, 1], [0, -1], [0, -1]])
    data = Data(np.ones((2, 8)), normalise=False)
    with patch("pyspi.statistics.spectral.spectral_workspace", return_value=conn):
        # First bin has no nonzero imaginary observations; second is valid.
        value = cls(fmin=.25, fmax=.5, statistic=statistic).multivariate(data)
        expected = -.2 if cls is spectral.DebiasedSquaredWeightedPhaseLagIndex else .2
        np.testing.assert_allclose(value[0, 1], expected)
        with pytest.warns(RuntimeWarning, match="(Mean of empty slice|All-NaN slice)"):
            missing = cls(fmin=.25, fmax=.25, statistic=statistic).multivariate(data)
        assert np.isnan(missing).all()


@pytest.mark.parametrize("tapers", [5, 256])
def test_nonzero_count_is_compact_and_does_not_overflow(tapers):
    conn = connection(np.ones((tapers, 1)))
    moments = conn._phase_statistics("sign")
    assert moments["nonzero"][0, 1, 0, 1] == tapers
    assert moments["nonzero"].dtype.itemsize == (1 if tapers == 5 else 2)
    assert conn.measure("debiased_squared_phase_lag_index")[0, 1, 0, 1] == 1


@pytest.mark.parametrize("small", [1e-10, 1e-20, 1e-100, 1e-250])
@pytest.mark.parametrize("sign", [-1, 1])
def test_tiny_second_weight_does_not_disappear_from_debiased_denominator(small, sign):
    # The squared-sum subtraction rounds to zero for small <= epsilon. There
    # is still one usable observation pair, whose product has the given sign.
    conn = connection(np.array([1, sign*small, 0, 0, 0])[:, None])
    actual = conn.measure("debiased_squared_weighted_phase_lag_index")[0, 1, 0, 1]
    np.testing.assert_allclose(actual, sign, rtol=1e-13)


def test_phase_pair_tiles_match_ordered_scalar_formulas_and_channel_permutations():
    rng = np.random.default_rng(84)
    coefficients = rng.normal(size=(2, 2, 3, 8, 35)) + 1j*rng.normal(size=(2, 2, 3, 8, 35))
    frequencies = np.fft.fftfreq(8)
    conn = SpectralConnectivity(coefficients, frequencies, np.arange(2.))
    order = rng.permutation(35)
    shuffled = SpectralConnectivity(coefficients[..., order], frequencies, np.arange(2.))
    names = ["phase_locking_value", "pairwise_phase_consistency", "signed_phase_lag_index",
             "phase_lag_index", "weighted_phase_lag_index", "signed_weighted_phase_lag_index",
             "debiased_squared_phase_lag_index", "debiased_squared_weighted_phase_lag_index"]
    values = {name: conn.measure(name) for name in names}
    for name in names:
        np.testing.assert_allclose(shuffled.measure(name), values[name][..., order, :][..., order],
                                   rtol=1e-12, atol=1e-14)
    # Include both orientations and pairs spanning the 32-channel tile boundary.
    for i, j in [(0, 34), (34, 0), (33, 2), (31, 32)]:
        for w in range(2):
            cross = (coefficients[w, :, :, 2, i]*coefficients[w, :, :, 2, j].conj()).ravel()
            y = cross.imag
            signs = np.sign(y)
            unit = cross/abs(cross)
            pairs = [(a, b) for a in range(len(y)) for b in range(len(y)) if a != b]
            expected = {
                "phase_locking_value": abs(unit.mean()),
                "pairwise_phase_consistency": np.mean([(unit[a]*unit[b].conjugate()).real for a,b in pairs]),
                "signed_phase_lag_index": signs.mean(), "phase_lag_index": abs(signs.mean()),
                "weighted_phase_lag_index": abs(y.sum())/abs(y).sum(),
                "signed_weighted_phase_lag_index": y.sum()/abs(y).sum(),
                "debiased_squared_phase_lag_index": np.mean([signs[a]*signs[b] for a,b in pairs]),
                "debiased_squared_weighted_phase_lag_index": (
                    sum(y[a]*y[b] for a,b in pairs)/sum(abs(y[a]*y[b]) for a,b in pairs)),
            }
            for name in names:
                np.testing.assert_allclose(values[name][w, 2, i, j], expected[name], atol=1e-14)
