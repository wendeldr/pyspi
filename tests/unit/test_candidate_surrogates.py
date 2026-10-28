"""Check the actual invariants and reference artifact behind the pilot controls."""

import importlib.util
from pathlib import Path

import numpy as np

from pyspi.workflow._reference import apply_shaft_reference, reference_plan


spec = importlib.util.spec_from_file_location("candidate_assessment", Path(__file__).resolve().parents[1] / "candidate_pilot/assess.py")
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


def test_phase_control_preserves_each_views_periodogram():
    rng = np.random.default_rng(8)
    values = [rng.normal(size=(4, 1024)) for _ in range(3)]
    result = assessment.surrogate(values, rng, "phase")
    for before, after in zip(values, result):
        np.testing.assert_allclose(abs(np.fft.rfft(before)), abs(np.fft.rfft(after)), rtol=1e-12, atol=1e-12)
    assert not np.allclose(result[0], values[0])


def test_shift_control_preserves_samples_autocorrelation_and_view_alignment():
    rng = np.random.default_rng(19)
    base = rng.normal(size=(4, 1024))
    first, second = assessment.surrogate([base, 3*base], rng, "shift")
    np.testing.assert_array_equal(np.sort(first), np.sort(base))
    np.testing.assert_allclose(abs(np.fft.rfft(first)), abs(np.fft.rfft(base)), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(second, 3*first)


def test_local_reference_creates_expected_coupling_from_independent_sources():
    names = [f"A{i}" for i in range(1, 6)]
    reference = {"method": "laplacian", "shafts": {"A": names}}
    plan = reference_plan(names, reference)
    original = np.random.default_rng(4).normal(size=(5, 32768))
    before = assessment.surrogate([original], np.random.default_rng(10), "shift")[0]
    referenced = apply_shaft_reference(original, names, reference, plan)
    reference_after_shift = apply_shaft_reference(before, names, reference, plan)
    shifted_after_reference = assessment.surrogate([referenced], np.random.default_rng(10), "shift")[0]
    assert abs(np.corrcoef(reference_after_shift[1:3])[0, 1] + 2/3) < .03
    assert abs(np.corrcoef(shifted_after_reference[1:3])[0, 1]) < .03
