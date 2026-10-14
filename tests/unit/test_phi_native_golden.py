"""Golden and reference checks for ``pyspi.lib.phi_native``.

``phi_comp`` values are locked against ``tests/regression/goldens/phi_native.npz``
(written by ``generate_goldens.py --phi``) with the ``phi_*`` policy tolerance.
"""
import json

import numpy as np
import pytest

from pyspi.lib.phi_native import cov_comp, logdet, phi_comp
from regression import harness, spi_policy

PHI_GOLDEN = harness.GOLDEN_DIR / "phi_native.npz"


def _load_phi_golden():
    if not PHI_GOLDEN.is_file():
        pytest.skip(f"missing {PHI_GOLDEN}: run tests/regression/generate_goldens.py --phi")
    with np.load(PHI_GOLDEN) as z:
        return {k: z[k] for k in z.files}


def test_phi_comp_matches_golden():
    gold = _load_phi_golden()
    meta = json.loads(str(gold["params_json"]))
    X, Z = gold["X"], gold["Z"]
    params = dict(meta["params"])
    keys = sorted(k for k in gold if k.startswith("phi_"))
    assert keys, "phi_native.npz holds no phi_* values"
    assert set(keys) == set(meta["keys"])

    rtol, atol = spi_policy.tolerance_for("phi_star")
    for key in keys:
        opts = meta["keys"][key]
        options = {"type_of_phi": opts["type_of_phi"], "type_of_dist": meta.get("type_of_dist", "Gauss"),
                   "normalization": opts["normalization"]}
        got = phi_comp(X.copy(), Z.copy(), dict(params), options)
        np.testing.assert_allclose(got, gold[key], rtol=rtol, atol=atol, err_msg=key)


def test_logdet_matches_slogdet_for_spd_matrix():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 4))
    M = A @ A.T + 4 * np.eye(4)
    sign, expected = np.linalg.slogdet(M)
    assert sign == 1.0
    assert logdet(M) == pytest.approx(expected, rel=1e-12)


def test_cov_comp_blocks_for_two_variables():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2, 60))

    joint = cov_comp(X, tau=1)
    assert set(joint) == {"Cov_X", "Cov_Y", "Cov_XY"}
    past, present = X[:, :-1], X[:, 1:]
    np.testing.assert_allclose(joint["Cov_X"], np.cov(past), rtol=1e-12)
    np.testing.assert_allclose(joint["Cov_Y"], np.cov(present), rtol=1e-12)
    np.testing.assert_allclose(joint["Cov_XY"], np.cov(past, present)[:2, 2:], rtol=1e-12)

    single = cov_comp(X, tau=1, isjoint=False)
    assert set(single) == {"Cov_X"}
    np.testing.assert_allclose(single["Cov_X"], np.cov(X), rtol=1e-12)
