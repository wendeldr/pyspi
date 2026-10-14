"""Adversarial checks of the change detectors themselves, without SPI math."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from regression import generate_goldens as generator, harness


MATRIX = np.array([[np.nan, -0.25], [0.75, np.nan]])


@pytest.mark.parametrize("change", ["scale", "transpose", "nan", "inf", "inf_sign", "shape", "complex", "object"])
def test_comparator_rejects_output_mutations(change):
    expected = MATRIX.copy()
    actual = expected.copy()
    if change == "scale":
        actual *= 1.1
    elif change == "transpose":
        actual = actual.T
    elif change == "nan":
        actual[0, 1] = np.nan
    elif change == "inf":
        actual[0, 1] = np.inf
    elif change == "inf_sign":
        expected[0, 1], actual[0, 1] = np.inf, -np.inf
    elif change == "shape":
        actual = actual[:1]
    elif change == "complex":
        actual = actual.astype(complex)
        actual[0, 1] += 2j
    else:
        actual = actual.astype(object)
    with pytest.raises(AssertionError):
        harness.assert_spi_allclose(actual, expected, "test_spi")


def test_comparator_preserves_nonfinite_masks_and_explicit_tolerances():
    expected = np.array([[np.nan, np.inf], [-np.inf, 1.0]])
    harness.assert_spi_allclose(expected.copy(), expected, "test_spi")
    actual = expected.copy()
    actual[1, 1] += 1e-7
    harness.assert_spi_allclose(actual, expected, "test_spi")
    with pytest.raises(AssertionError):
        harness.assert_spi_allclose(actual, expected, "test_spi", rtol=0, atol=0)


@pytest.mark.parametrize("actual,expected", [([np.nan], [1.0]), ([np.inf], [-np.inf]),
    ([1j], [0.0]), ([[1.0]], [1.0])])
def test_difference_diagnostics_do_not_report_zero_for_invalid_outputs(actual, expected):
    assert harness.max_differences(actual, expected) == (np.inf, np.inf)


def test_difference_diagnostics_report_relative_error_against_zero():
    assert harness.max_differences([1.0], [0.0]) == (1.0, np.inf)
    assert harness.max_differences([np.nan, np.inf], [np.nan, np.inf]) == (0.0, 0.0)


@pytest.mark.parametrize("available,required", [(False, False), (False, True), (True, True)])
def test_complete_run_can_require_java_instead_of_silently_skipping(monkeypatch, available, required):
    import conftest
    monkeypatch.setattr(harness, "ensure_jvm", lambda: available)
    config = SimpleNamespace(getoption=lambda name: required)
    if required and not available:
        with pytest.raises(pytest.fail.Exception, match="JIDT JVM is required"):
            conftest.jvm.__wrapped__(config)
    else:
        assert conftest.jvm.__wrapped__(config) is available


def _fake_spi(monkeypatch, outcome):
    def multivariate(data):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    spi = SimpleNamespace(identifier="toy", multivariate=multivariate)
    monkeypatch.setattr(harness, "build_spi", lambda entry: spi)
    return spi


@pytest.mark.parametrize("outcome", [IndexError("different message"), RuntimeError("recorded"),
                                     np.full((2, 2), np.nan), MATRIX])
def test_recorded_error_cannot_hide_another_failure_or_a_success(monkeypatch, outcome):
    _fake_spi(monkeypatch, outcome)
    entry = {"status": "error", "error": "IndexError('recorded')"}
    with pytest.raises(AssertionError):
        harness.compute_recorded(entry, SimpleNamespace(n_processes=2), 42, lambda seed: None)


def test_recorded_error_matches_exception_identity(monkeypatch):
    _fake_spi(monkeypatch, IndexError("recorded"))
    entry = {"status": "error", "error": "IndexError('recorded')"}
    got = harness.compute_recorded(entry, SimpleNamespace(n_processes=2), 42, lambda seed: None)
    assert got.shape == (2, 2) and np.isnan(got).all()


@pytest.mark.parametrize("status", ["ok", "partial_nan", "all_nan"])
def test_success_snapshot_does_not_swallow_exceptions(monkeypatch, status):
    _fake_spi(monkeypatch, RuntimeError("new calculation failure"))
    with pytest.raises(RuntimeError, match="new calculation failure"):
        harness.compute_recorded({"status": status}, SimpleNamespace(n_processes=2), 42, lambda seed: None)


def test_recorded_status_changes_are_detected(monkeypatch):
    _fake_spi(monkeypatch, np.full((2, 2), np.nan))
    with pytest.raises(AssertionError, match="status changed"):
        harness.compute_recorded({"status": "ok"}, SimpleNamespace(n_processes=2), 42, lambda seed: None)


@pytest.mark.parametrize("stage", ["constructor", "seed"])
def test_setup_failures_never_become_recorded_calculation_errors(monkeypatch, stage):
    _fake_spi(monkeypatch, IndexError("recorded"))
    def boom(*args):
        raise RuntimeError("setup defect")
    if stage == "constructor":
        monkeypatch.setattr(harness, "build_spi", boom)
    with pytest.raises(RuntimeError, match="setup defect"):
        harness.compute_recorded({"status": "error", "error": "IndexError('recorded')"},
                                 SimpleNamespace(n_processes=2), 42,
                                 boom if stage == "seed" else lambda seed: None)


@pytest.mark.parametrize("outcome", [np.ones((2, 1)), np.ones((3, 3)), np.ones((2, 2), dtype=complex)])
def test_compute_rejects_invalid_shape_and_complex_results(monkeypatch, outcome):
    spi = _fake_spi(monkeypatch, outcome)
    with pytest.raises(AssertionError):
        harness.compute_spi(spi, SimpleNamespace(n_processes=2))


def test_postprocessing_does_not_mutate_the_spi_cache(monkeypatch):
    original = np.arange(4.0).reshape(2, 2)
    spi = _fake_spi(monkeypatch, original)
    got = harness.compute_spi(spi, SimpleNamespace(n_processes=2))
    np.testing.assert_array_equal(original, [[0, 1], [2, 3]])
    np.testing.assert_array_equal(got, [[np.nan, 1], [2, np.nan]])


@pytest.mark.parametrize("missing", ["meta.json", "values.npz"])
def test_missing_required_golden_cannot_silently_skip(monkeypatch, tmp_path, missing):
    monkeypatch.setattr(harness, "GOLDEN_DIR", tmp_path)
    folder = tmp_path / "small"
    folder.mkdir()
    (folder / ({"meta.json", "values.npz"} - {missing}).pop()).touch()
    with pytest.raises(FileNotFoundError, match="required golden set"):
        harness.require_golden("small")


@pytest.fixture
def saved_set(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "GOLDEN_DIR", tmp_path)
    record = dict(module=".statistics.basic", cls="Toy", params={"scale": 1},
                  deps=[], labels=["signed"], status="ok", error=None, time_s=0.01)
    meta = dict(spis={"toy": record}, dataset={"file": "toy.npy"}, seed=42,
                calculator_order=["toy"])
    values = {"toy": MATRIX.copy()}
    harness.save_golden("toy", values, meta)
    return values, meta


@pytest.mark.parametrize("field", ["dataset", "seed", "params", "labels", "values", "error"])
def test_generator_check_detects_metadata_and_value_changes(saved_set, field):
    values, meta = saved_set
    changed = copy.deepcopy(meta)
    if field == "dataset":
        changed["dataset"]["file"] = "different.npy"
    elif field == "seed":
        changed["seed"] = 999
    elif field == "params":
        changed["spis"]["toy"]["params"]["scale"] = 2
    elif field == "labels":
        changed["spis"]["toy"]["labels"] = ["unsigned"]
    elif field == "values":
        values = {"toy": MATRIX * 2}
    else:
        values = {"toy": np.full((2, 2), np.nan)}
        for target, error in ((meta, "IndexError('original')"), (changed, "RuntimeError('new')")):
            target["spis"]["toy"].update(status="error", error=error)
        harness.save_golden("toy", values, meta)
    assert not generator.check_set("toy", values, changed, partial=False)


@pytest.mark.parametrize("field", ["dataset", "seed"])
def test_partial_regeneration_refuses_to_mix_inputs(saved_set, field):
    values, meta = saved_set
    changed = copy.deepcopy(meta)
    changed[field] = 999 if field == "seed" else {"file": "different.npy"}
    before = {p.name: p.read_bytes() for p in harness.golden_dir("toy").iterdir()}
    with pytest.raises(ValueError, match=f"different {field}"):
        generator.merge_into_existing("toy", values, changed, partial=True)
    assert before == {p.name: p.read_bytes() for p in harness.golden_dir("toy").iterdir()}


def test_phi_check_does_not_ignore_parameter_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "GOLDEN_DIR", tmp_path)
    desc = dict(seed=42, Z=[1, 2], params={"tau": 1}, type_of_dist="Gauss", keys={"phi_star": {}})
    values = {"phi_star": np.array(0.5), "params_json": np.array(json.dumps(desc))}
    np.savez_compressed(tmp_path / generator.PHI_FILE, **values)
    desc["params"]["tau"] = 99
    changed = dict(values, params_json=np.array(json.dumps(desc)))
    monkeypatch.setattr(generator, "phi_values", lambda: changed)
    assert not generator.do_phi(check=True)


def test_repeat_audit_detects_different_exception_causes(monkeypatch):
    errors = iter([IndexError("original"), RuntimeError("new")])
    def compute(*args):
        raise next(errors)
    monkeypatch.setattr(harness, "compute_spi", compute)
    monkeypatch.setattr(harness, "make_data", lambda *args: None)
    monkeypatch.setattr(harness, "build_spi", lambda *args: None)
    monkeypatch.setattr(harness, "seed_all", lambda *args: None)
    _, _, problem = generator.run_one({"identifier": "toy"}, np.zeros((2, 10)), {}, repeat=2)
    assert problem is not None and "error state differs" in problem


@pytest.mark.parametrize("check", [False, True])
def test_nondeterministic_repeats_cannot_report_success_or_write(monkeypatch, saved_set, check):
    values, meta = saved_set
    monkeypatch.setattr(harness, "ensure_jvm", lambda: False)
    monkeypatch.setattr(harness, "import_statistics_modules", lambda: {})
    monkeypatch.setattr(harness, "optional_dependencies", lambda: {})
    monkeypatch.setattr(harness, "catalogue", lambda *args: ([], {"toy": {}}))
    monkeypatch.setattr(generator, "build_size_set", lambda *args: (values, meta, {"toy": "repeat differs"}))
    monkeypatch.setattr(generator, "summarise", lambda *args: None)
    monkeypatch.setattr(generator, "check_set", lambda *args: True)
    writes = []
    monkeypatch.setattr(harness, "save_golden", lambda *args: writes.append(args))
    args = ["--size", "small"] + (["--check"] if check else [])
    if check:
        assert generator.main(args) == 1
    else:
        with pytest.raises(ValueError, match="non-reproducible"):
            generator.main(args)
    assert writes == []
