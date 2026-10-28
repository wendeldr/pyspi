"""The Cholesky FAST-MCD reproduces scikit-learn's seeded estimates."""

import warnings

import numpy as np
import pytest
from sklearn.covariance import MinCovDet

from pyspi._robust import FastMinCovDet, _c_step, _distances, _factor


def datasets():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(1024, 12))
    outliers = clean.copy()
    outliers[:60] += 8
    correlated = rng.normal(size=(700, 6)) @ rng.normal(size=(6, 6))
    small = rng.normal(size=(200, 5))
    small[:15] *= 6
    return {"clean-1024x12": clean, "outliers-1024x12": outliers,
            "correlated-700x6": correlated, "small-200x5": small,
            "univariate-300x1": rng.normal(size=(300, 1))}


@pytest.mark.parametrize("name,X", datasets().items(), ids=list(datasets()))
@pytest.mark.parametrize("seed", [0, 42])
def test_fast_min_cov_det_matches_sklearn_with_the_same_seed(name, X, seed):
    reference = MinCovDet(random_state=seed).fit(X)
    actual = FastMinCovDet(random_state=seed).fit(X)
    np.testing.assert_array_equal(actual.raw_support_, reference.raw_support_)
    np.testing.assert_array_equal(actual.support_, reference.support_)
    np.testing.assert_allclose(actual.covariance_, reference.covariance_, rtol=1e-12, atol=0)
    np.testing.assert_allclose(actual.precision_, reference.precision_, rtol=1e-10, atol=0)
    np.testing.assert_allclose(actual.location_, reference.location_, rtol=1e-12, atol=0)


def test_c_step_distances_and_log_determinant_match_the_dense_formulas():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))
    covariance = np.cov(X, rowvar=False)
    factor, logdet = _factor(covariance)
    assert factor[0] == "cholesky"
    assert logdet == pytest.approx(np.linalg.slogdet(covariance)[1], rel=1e-12)
    centered = X - X.mean(0)
    dense = (centered @ np.linalg.inv(covariance) * centered).sum(1)
    np.testing.assert_allclose(_distances(factor, centered), dense, rtol=1e-11)


def test_singular_candidate_falls_back_to_the_pseudo_inverse_step():
    singular = np.ones((3, 3))
    factor, logdet = _factor(singular)
    assert factor[0] == "pinvh" and logdet == -np.inf
    X = np.repeat(np.arange(6.)[:, None], 3, axis=1)   # rank one: every support is singular
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        location, covariance, det, support, dist = _c_step(X, 4, np.random.RandomState(0))
    assert det == -np.inf and support.sum() == 4
    assert np.isfinite(dist).all()


@pytest.mark.parametrize("seed", [1, 3, 7])
def test_car_fit_preserves_reference_support_and_values(seed):
    # CAR followed by per-channel z-scoring remains rank deficient even though
    # its null vector is no longer the all-ones vector. Seed 1 exposed a 3%
    # covariance change from accidentally positive Cholesky pivots.
    X = np.random.default_rng(seed).normal(size=(256, 8))
    X -= X.mean(axis=1, keepdims=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        reference = MinCovDet(random_state=42).fit(X)
        actual = FastMinCovDet(random_state=42).fit(X)
    np.testing.assert_array_equal(actual.raw_support_, reference.raw_support_)
    np.testing.assert_array_equal(actual.support_, reference.support_)
    np.testing.assert_array_equal(actual.covariance_, reference.covariance_)
    np.testing.assert_array_equal(actual.precision_, reference.precision_)
