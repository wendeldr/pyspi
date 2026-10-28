"""Reference geometry tests require neither EEG samples nor connectivity."""

from pathlib import Path

import numpy as np
import pytest

from pyspi.workflow._config import WorkflowError, reference_config
from pyspi.workflow._reference import apply_shaft_reference, available_indices, reference_details, reference_plan


def test_laplacian_preserves_physical_gaps_and_order():
    names = ["A5", "A2", "A1", "A4", "A6", "A7"]
    cfg = reference_config({"method": "laplacian", "shafts": {"A": [f"A{i}" for i in range(1, 8)]}}, Path("."))
    weights, available, reasons = plan = reference_plan(names, cfg)
    assert available.tolist() == [True, False, False, False, True, False]
    assert reasons[1] == reasons[3] == "missing_adjacent_contact"
    assert reasons[2] == reasons[-1] == "shaft_endpoint"
    source = np.random.default_rng(4).normal(size=(6, 50))
    expected = np.zeros_like(source)
    expected[0] = source[0] - (source[3] + source[4]) / 2
    expected[4] = source[4] - (source[0] + source[5]) / 2
    np.testing.assert_allclose(weights @ source, expected)
    details = reference_details(names, plan)
    np.testing.assert_array_equal(available_indices({"reference_geometry": details}, 6), [0, 4])
    assert details["coefficients"]["A2"] == {}
    assert details["shared_source_pairs"][0]["channels"] == ["A5", "A6"]
    assert details["shared_source_pairs"][0]["independent_equal_variance_correlation"] == pytest.approx(-2/3)


def test_reference_geometry_survives_channel_permutation():
    names = [f"A{i}" for i in range(1, 7)]
    cfg = {"method": "laplacian", "shafts": {"A": names}}
    weights, available, _ = reference_plan(names, cfg)
    order = np.array([4, 0, 3, 5, 1, 2])
    permuted, perm_available, _ = reference_plan([names[i] for i in order], cfg)
    np.testing.assert_array_equal(permuted, weights[np.ix_(order, order)])
    np.testing.assert_array_equal(perm_available, available[order])


def test_shaft_average_uses_only_retained_same_shaft_and_marks_singletons():
    names = ["A1", "B2", "A3", "C1"]
    cfg = {"method": "shaft_average", "shafts": {"A": ["A1", "A2", "A3"], "B": ["B1", "B2"], "C": ["C1"]}}
    plan = reference_plan(names, cfg)
    np.testing.assert_array_equal(plan[1], [True, False, True, False])
    assert reference_details(names, plan)["proportional_pairs"] == [["A1", "A3"]]
    np.testing.assert_array_equal(plan[0][0], [.5, 0, -.5, 0])


def test_car_shared_reference_null_and_proportional_two_contact_case():
    names = ["A", "B", "C", "D"]
    details = reference_details(names, reference_plan(names, {"method": "car"}))
    assert len(details["shared_source_pairs"]) == 6
    assert not details["proportional_pairs"]
    assert all(pair["independent_equal_variance_correlation"] == pytest.approx(-1/3)
               for pair in details["shared_source_pairs"])
    pair = reference_details(names[:2], reference_plan(names[:2], {"method": "car"}))
    assert pair["proportional_pairs"] == [["A", "B"]]


@pytest.mark.parametrize("shafts", [{}, {"A": []}, {"A": ["A1", "A1"]},
                                    {"A": ["A1"], "B": ["A1"]}, {"A": [None]}])
def test_invalid_layout_is_rejected(shafts):
    with pytest.raises(WorkflowError):
        reference_config({"method": "laplacian", "shafts": shafts}, Path("."))


def test_unassigned_retained_channel_is_an_error():
    with pytest.raises(WorkflowError, match="absent from explicit shaft layout"):
        reference_plan(["A1", "A2", "B1"], {"method": "laplacian", "shafts": {"A": ["A1", "A2"]}})


@pytest.mark.parametrize("method", ["laplacian", "shaft_average"])
def test_sparse_application_matches_full_linear_operator(method):
    names = ["B2", "A1", "A2", "A3", "A5", "A6", "B1", "B3"]
    cfg = {"method": method, "shafts": {"A": [f"A{i}" for i in range(1, 7)], "B": ["B1", "B2", "B3"]}}
    plan = reference_plan(names, cfg)
    source = np.random.default_rng(7).normal(size=(len(names), 2048))
    before = source.copy()
    actual = apply_shaft_reference(source, names, cfg, plan)
    np.testing.assert_allclose(actual, plan[0] @ source, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(source, before)
