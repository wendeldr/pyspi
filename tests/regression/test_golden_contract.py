"""Committed baselines must have complete, consistent metadata and payloads."""

import numpy as np
import pytest

from regression import harness, spi_policy
from regression.generate_goldens import catalogue_mismatches


@pytest.mark.parametrize("golden_set", ["small", "full", "preproc", "extra/small"])
def test_golden_corpus_is_complete_and_self_consistent(golden_set):
    harness.require_golden(golden_set)
    meta, values = harness.load_meta(golden_set), harness.load_values(golden_set)
    records = meta.get("variants", meta["spis"])
    assert values and set(values) == set(records)
    if golden_set == "preproc":
        assert set(meta["spis"]) <= set(meta["calculator_order"])
    else:
        assert set(meta["calculator_order"]) == set(meta["spis"])
    assert len(meta["calculator_order"]) == len(set(meta["calculator_order"]))
    assert meta["duplicate_identifiers"] == []
    if golden_set in ("small", "full"):
        # An accidentally broadened skip pattern must not disable numerical
        # checks of unrelated Calculator blocks while leaving tests green.
        approved_skips = {ident for ident, entry in meta["spis"].items()
                          if (entry["cls"] in ("Covariance", "Precision")
                              and (entry["params"] or {}).get("estimator") == "MinCovDet")
                          or (entry["cls"] == "Barycenter"
                              and (entry["params"] or {}).get("mode") == "sgddtw")}
        actual_skips = {ident for ident in meta["spis"]
                        if spi_policy.matches_any(ident, spi_policy.CALC_E2E_SKIP)}
        assert actual_skips == approved_skips
    n = harness.load_raw(meta["dataset"]).shape[0]
    for key, value in values.items():
        assert value.shape == (n, n), key
        assert value.dtype == np.dtype("float64"), key
        assert np.isnan(np.diag(value)).all(), key
        record = records[key]
        if record["status"] == "error":
            assert record["error"] and np.isnan(value).all(), key
        else:
            assert record["error"] is None, key
            assert harness.status_of(value) == record["status"], key


def test_preprocessing_corpus_covers_all_four_flags_and_current_catalogue():
    meta = harness.load_meta("preproc")
    expected_keys = {spi_policy.preproc_key(ident, normalise, detrend)
                     for ident in meta["spis"] for normalise, detrend in spi_policy.PREPROC_VARIANTS}
    assert set(meta["variants"]) == expected_keys
    main = harness.load_meta("small")
    assert set(meta["spis"]) == set(spi_policy.select_preproc_identifiers(main["calculator_order"]))
    for ident, entry in meta["spis"].items():
        assert not catalogue_mismatches(main["spis"][ident], entry), ident
