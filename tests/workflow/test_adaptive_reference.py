"""Independent formulas for the contact-preserving custom reference."""
from pathlib import Path

import numpy as np
import pytest

from pyspi.workflow._adaptive_reference import build_adaptive_reference, resolve_reference
from pyspi.workflow._config import WorkflowError, reference_config
from pyspi.workflow._reference import reference_plan, reference_details, apply_shaft_reference


def build(names, groups=None):
    return build_adaptive_reference(names, groups or ["A"] * len(names), patient_id="test")


def test_single_gap_and_endpoint_formula():
    names = ["A1", "A3", "A4"]
    ref = build(names)
    expected = [[1, -1, 0], [-1/3, 1, -2/3], [0, -1, 1]]
    plan = reference_plan(names, ref)
    np.testing.assert_allclose(plan[0], expected)
    assert plan[1].all()
    assert ref["rules"]["A1"]["rule"] == "bipolar"
    assert ref["rules"]["A3"]["rule"] == "weighted_laplacian"
    # A linear spatial trend has zero interior curvature despite the gap.
    np.testing.assert_allclose((plan[0] @ [1., 3., 4.])[1], 0, atol=1e-15)


def test_equal_gaps_and_large_gap_fallbacks():
    names = ["A1", "A3", "A5"]
    plan = reference_plan(names, build(names))
    np.testing.assert_array_equal(plan[0][1], [-.5, 1, -.5])
    names = ["A1", "A4", "A5", "A6"]
    ref = build(names)
    plan = reference_plan(names, ref)
    np.testing.assert_allclose(plan[0], [[.75, -.25, -.25, -.25],
                                       [-.25, .75, -.25, -.25],
                                       [0, -.5, 1, -.5], [0, 0, -1, 1]])
    assert [ref["rules"][n]["rule"] for n in names] == ["shaft_average", "shaft_average", "laplacian", "bipolar"]


def test_small_shafts_use_global_car_including_singletons():
    names = ["A1", "A4", "B1", "C1", "C2", "C3"]
    ref = build(names, ["A", "A", "B", "C", "C", "C"])
    actual = reference_plan(names, ref)
    car = reference_plan(names, {"method": "car"})
    np.testing.assert_allclose(actual[0][:3], car[0][:3])
    assert actual[1].all()
    singleton = reference_plan(["A1"], build(["A1"]))
    assert singleton[1].tolist() == [False]
    assert singleton[2] == ["zero_reference_row"]
    # With only two contacts in the entire recording even CAR is proportional.
    pair = reference_details(["A1", "A2"], reference_plan(["A1", "A2"], build(["A1", "A2"])))
    assert pair["proportional_pairs"] == [["A1", "A2"]]


def test_names_and_permutation_preserve_authoritative_axis():
    names = ["A1'10", "A1'8", "A1'9", "RPI2", "RPI1", "RPI3"]
    groups = ["A1'"] * 3 + ["RPI"] * 3
    ref = build(names, groups)
    assert ref["shafts"]["A1'"] == ["A1'8", "A1'9", "A1'10"]
    weights = reference_plan(names, ref)[0]
    order = [5, 1, 3, 0, 4, 2]
    perm_names = [names[i] for i in order]
    perm_ref = build(perm_names, [groups[i] for i in order])
    np.testing.assert_allclose(reference_plan(perm_names, perm_ref)[0], weights[np.ix_(order, order)])


def test_application_matches_formulas_cancels_shared_signal_and_preserves_input():
    names = ["A1", "A4", "A5", "A6", "B1", "B3", "B4", "C1"]
    ref = build(names, ["A"] * 4 + ["B"] * 3 + ["C"])
    plan = reference_plan(names, ref)
    source = np.random.default_rng(5).normal(size=(len(names), 300))
    before = source.copy()
    actual = apply_shaft_reference(source, names, ref, plan)
    np.testing.assert_allclose(actual, plan[0] @ source, atol=2e-15)
    shared = np.linspace(-10, 10, source.shape[1])
    np.testing.assert_allclose(apply_shaft_reference(source + shared, names, ref, plan), actual, atol=5e-15)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("names,groups", [(["A1", "A01"], ["A", "A"]),
    (["A1", "A1"], ["A", "A"]), (["A1"], ["B"]), (["A0"], ["A"]),
    (["A1x"], ["A"]), (["A1"], [""]), (["A1"], [])])
def test_ambiguous_or_inconsistent_numbering_fails(names, groups):
    with pytest.raises(WorkflowError):
        build_adaptive_reference(names, groups, patient_id="test")


def test_config_has_no_patient_specific_lists_and_requires_dataset():
    from types import SimpleNamespace
    ref = reference_config("adaptive_local", Path("."))
    assert ref == {"method": "adaptive_local"}
    with pytest.raises(WorkflowError, match="requires dataset"):
        resolve_reference(SimpleNamespace(reference=ref, dataset_directory=None), SimpleNamespace(patient_id="test"))
    with pytest.raises(WorkflowError):
        reference_config({"method": "adaptive_local", "shafts": {"A": ["A1"]}}, Path("."))
