import csv
import json
from pathlib import Path
import random

import numpy as np
import pytest

from pyspi.workflow import WorkflowError
from pyspi.workflow._metrics import resolve_metrics


def test_defaults_and_configured_order(metric_config):
    path = metric_config({"Covariance": {"configs": [{"squared": True}, {}]},
                          "CrossCorrelation": {"configs": None}})
    metrics = resolve_metrics(path)
    assert [m.identifier for m in metrics] == ["cov_sq_empirical", "cov_empirical",
                                             "xcorr_holm_max_maxlag0p125_alpha0p05"]
    assert metrics[0].parameters == {"squared": True}
    assert "EmpiricalCovariance" in metrics[0].constructor_signature


@pytest.mark.parametrize("fs", [None, 2048])
def test_xcorr_uses_workflow_sampling_rate(metric_config, fs):
    params = {"statistic": "min", "sigonly": False}
    if fs is not None:
        params["fs"] = fs
    path = metric_config({"CrossCorrelation": {"configs": [params]}})
    metric = resolve_metrics(path, sampling_rate_hz=2048)[0]
    assert metric.parameters["fs"] == 2048
    assert metric.identifier == "xcorr_min_maxlag0125ms"


def test_xcorr_conflicting_workflow_sampling_rate_rejected(metric_config):
    path = metric_config({"CrossCorrelation": {"configs": [{"fs": 1000}]}})
    with pytest.raises(WorkflowError, match="fs differs"):
        resolve_metrics(path, sampling_rate_hz=2048)


def test_spectral_hz_bands_must_match_workflow_sampling_rate(metric_config):
    path = metric_config({"CoherenceMagnitude": {"configs": [{"fs": 2048, "fmin": 30, "fmax": 70}, {"fs": 1}]}},
                         module=".statistics.spectral")
    assert [m.identifier for m in resolve_metrics(path, sampling_rate_hz=2048)] == [
        "cohmag_mean_band0030-0070hz", "cohmag_mean_band0-0p5cyc"]
    with pytest.raises(WorkflowError, match="fs differs"):
        resolve_metrics(path, sampling_rate_hz=1024)


@pytest.mark.parametrize("classes,match", [
    ({"SpearmanR": {"configs": [{"bogus": 1}, {"squared": True}]}}, "unknown parameters"),
    ({"Covariance": {"configs": [{"estimator": "Typo"}]}}, "unknown covariance"),
    ({"CrossCorrelation": {"configs": [{"statistic": "median"}]}}, "statistic must"),
    ({"Covariance": {"configs": [{"squared": "false"}]}}, "boolean"),
    ({"Covariance": {"configs": []}}, "nonempty list"),
    ({"Covariance": {"configs": [None]}}, "parameter configuration"),
    ({"Covariance": {"configs": [{"estimator": float("inf")}]}}, "finite JSON"),
    ({"Covariance": {"dependencies": ["not_a_dependency"], "configs": [{}]}}, "unknown optional"),
    ({"Covariance": {"dependencies": False, "configs": [{}]}}, "dependencies must"),
    ({"Covariance": {"dependencies": "java", "configs": [{}]}}, "dependencies must"),
    ({"Covariance": {"labels": "undirected", "configs": [{}]}}, "labels must"),
    ({"Covariance": {"extra": 1, "configs": [{}]}}, "unknown fields"),
    ({"Covariance": {"configs": [{}, {}]}}, "duplicate metric identifier"),
    ({"MissingSPI": {"configs": [{}]}}, "cannot initialize"),
    ({"_Private": {"configs": [{}]}}, "class name"),
])
def test_invalid_requests_fail_without_fallback(classes, match, metric_config):
    with pytest.raises(WorkflowError, match=match):
        resolve_metrics(metric_config(classes))


def test_ignored_kwargs_rejected(metric_config):
    with pytest.raises(WorkflowError, match="unknown parameters"):
        resolve_metrics(metric_config({"PairwiseDistance": {"configs": [{"metirc": "cosine"}]}},
                                     module=".statistics.distance"))


def test_unknown_distance_metric_rejected_before_fitting(metric_config):
    with pytest.raises(WorkflowError, match="unknown pairwise-distance"):
        resolve_metrics(metric_config({"PairwiseDistance": {"configs": [{"metric": "typo"}]}},
                                     module=".statistics.distance"))


@pytest.mark.java
def test_unknown_information_theory_estimator_rejected(metric_config):
    with pytest.raises(WorkflowError, match="unknown information-theory estimator"):
        resolve_metrics(metric_config({"MutualInfo": {"configs": [{"estimator": "typo"}]}},
                                     module=".statistics.infotheory"))


def test_missing_required_parameter_rejected(metric_config):
    with pytest.raises(WorkflowError, match="invalid constructor parameters"):
        resolve_metrics(metric_config({"LinearModel": {"configs": None}}, module=".statistics.misc"))


def test_constructor_domain_error_is_not_replaced_with_defaults(metric_config):
    with pytest.raises(WorkflowError, match="Unknown Barycenter mode"):
        resolve_metrics(metric_config({"Barycenter": {"configs": [{"mode": "typo"}]}},
                                     module=".statistics.distance"))


def test_java_is_required_not_silently_skipped(metric_config, monkeypatch):
    from pyspi import utils
    monkeypatch.setattr(utils, "is_jpype_jvm_available", lambda: False)
    with pytest.raises(WorkflowError, match="require an available Java"):
        resolve_metrics(metric_config({"Covariance": {"dependencies": ["java"], "configs": [{}]}}))


def test_jidt_java_dependency_inferred_if_omitted(metric_config, monkeypatch):
    from pyspi import utils
    monkeypatch.setattr(utils, "is_jpype_jvm_available", lambda: False)
    with pytest.raises(WorkflowError, match="require an available Java"):
        resolve_metrics(metric_config({"MutualInfo": {"configs": [{"estimator": "kraskov"}]}}, module=".statistics.infotheory"))


def test_native_gaussian_resolves_without_java(metric_config, monkeypatch):
    from pyspi import utils
    def forbidden():
        raise AssertionError("Native Gaussian resolution must not check/start Java")
    monkeypatch.setattr(utils, "is_jpype_jvm_available", forbidden)
    metrics = resolve_metrics(metric_config({"MutualInfo": {"dependencies": ["java"], "configs": [{}]}},
                                            module=".statistics.infotheory"))
    assert metrics[0].identifier == "mi_gaussian" and metrics[0].dependencies == ()


@pytest.mark.parametrize("params", [{"fs": 0}, {"fmin": -1}, {"fmin": .4, "fmax": .3},
                                    {"fmax": .6}, {"fs": True}, {"fmax": "0.2"}])
def test_invalid_spectral_geometry_rejected(params, metric_config):
    with pytest.raises(WorkflowError):
        resolve_metrics(metric_config({"CoherenceMagnitude": {"configs": [params]}}, module=".statistics.spectral"))


def test_parameters_keep_full_precision_and_close_bands_stay_distinct(metric_config):
    params = {"fmin": 0.1000123456789, "fmax": 0.4}
    path = metric_config({"CoherenceMagnitude": {"configs": [params]}}, module=".statistics.spectral")
    metric = resolve_metrics(path)[0]
    assert metric.parameters == params
    path = metric_config({"CoherenceMagnitude": {"configs": [params, {"fmin": 0.100023456789, "fmax": 0.4}]}},
                         module=".statistics.spectral")
    assert [m.identifier for m in resolve_metrics(path)] == [
        "cohmag_mean_band0p1000123456789-0p4cyc", "cohmag_mean_band0p100023456789-0p4cyc"]


def test_rng_and_shared_label_state_restored_even_on_failure(metric_config):
    from pyspi.statistics.basic import SpearmanR
    labels = SpearmanR.labels
    original = labels.copy()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    path = metric_config({"SpearmanR": {"configs": [{"squared": True}, {"squared": True}]}})
    with pytest.raises(WorkflowError, match="duplicate"):
        resolve_metrics(path)
    assert SpearmanR.labels is labels and labels == original
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and np.array_equal(actual[1], numpy_state[1]) and actual[2:] == numpy_state[2:]
    assert random.getstate() == python_state


@pytest.mark.java
def test_reviewed_preset_preserves_unrelated_inventory_and_applies_approved_changes():
    import pyspi
    path = Path(pyspi.__file__).parent / "workflow/reviewed.yaml"
    metrics = resolve_metrics(path)
    assert len(metrics) == len({m.identifier for m in metrics}) == 201
    additions = {m.identifier for m in resolve_metrics(path.parent / "additional_candidates_car.yaml")}
    assert len(additions) == 13
    assert additions <= {m.identifier for m in metrics}
    root = Path(__file__).resolve().parents[2]
    with open(root / "docs/metric_names.csv") as f:
        rows = list(csv.DictReader(f))
    # Prefer the preset's own mapping: its bands are in Hz at 2048 Hz.
    renamed = {row["old_identifier"]: row["new_identifier"] for row in rows}
    renamed.update({row["old_identifier"]: row["new_identifier"] for row in rows
                    if "pyspi/workflow/reviewed.yaml" in row["configs"].split(";")})
    for patient in ("004", "105", "111"):
        survey = json.loads((root / f"docs/storage_comparison/explore/properties_{patient}_64.json").read_text())
        historical = {renamed.get(row["metric"], row["metric"]) for row in survey["rows"]}
        changed_classes = {"CrossCorrelation", "SignedPhaseLagIndex", "SignedWeightedPhaseLagIndex"}
        removed = {"bary_euclidean_mean", "bary_sq_euclidean_mean", "lmfit_ridge_penaltysum_alpha1", "lmfit_SGDRegressor",
                   "je_gaussian", "ce_gaussian", "lmfit_lasso_alpha1",
                   "te_symbolic_k-1_kt-1_l-1_lt-1", "te_symbolic_k-10_kt-1_l-1_lt-1",
                   "cov_empirical", "cov_shrunk", "cov_sq_shrunk",
                   "prec_ellipticenvelope", "prec_sq_ellipticenvelope",
                   "cov_sq_graphicallasso", "prec_graphicallasso", "prec_sq_graphicallasso",
                   "cov_graphicallassocv", "cov_sq_graphicallassocv",
                   "prec_graphicallassocv", "prec_sq_graphicallassocv"}
        assert {m.identifier for m in metrics if m.class_name not in changed_classes} - additions == {
            identifier for identifier in historical if not identifier.startswith(("xcorr", "gd_"))
            and identifier not in removed} | {"lmfit_lasso_alpha0p1",
                "te_symbolic_patternlength03_spacing04_shift04", "te_symbolic_patternlength03_spacing16_shift16",
                "gc_gaussian_targethistory04_targetdelay01_sourcehistory04_sourcedelay01"}
        assert {m.identifier for m in metrics if m.class_name == "CrossCorrelation"} == {
            f"xcorr{sq}_{stat}_maxlag0p125"
            for sq in ("", "_sq") for stat in ("max", "mean")} | {
            f"xcorr_{stat}_maxlag0p125" for stat in ("min", "peak")} | {
            f"xcorr{sq}_holm_{stat}_maxlag0p125_alpha0p05"
            for stat, sq in (("max", ""), ("max", "_sq"), ("mean", ""), ("mean", "_sq"),
                             ("min", ""), ("peak", ""))}
    for original, signed, prefix in (("PhaseLagIndex", "SignedPhaseLagIndex", "pli"),
                                     ("WeightedPhaseLagIndex", "SignedWeightedPhaseLagIndex", "wpli")):
        expected = {m.identifier.replace(prefix+"_", prefix+"_signed_", 1)
                    for m in metrics if m.class_name == original}
        assert len(expected) == 11
        assert {m.identifier for m in metrics if m.class_name == signed} == expected
    assert not {m.class_name for m in metrics} & {
        "GroupDelay", "DirectedCoherence", "PartialDirectedCoherence", "GeneralizedPartialDirectedCoherence",
        "DirectedTransferFunction", "DirectDirectedTransferFunction", "SpectralGrangerCausality"}


@pytest.mark.java
@pytest.mark.parametrize("filename,count", [("config.yaml", 298), ("fast_config.yaml", 230),
                                           ("sonnet_config.yaml", 14), ("fabfour_config.yaml", 4)])
def test_existing_presets_resolve_without_computation(filename, count):
    import pyspi
    metrics = resolve_metrics(Path(pyspi.__file__).parent / filename)
    assert len(metrics) == len({m.identifier for m in metrics}) == count
    if filename == "config.yaml":
        golden = json.loads((Path(__file__).resolve().parents[1] / "regression/goldens/small/meta.json").read_text())
        assert [m.identifier for m in metrics] == golden["calculator_order"]


@pytest.mark.parametrize("content", ["{}", "[]", ".statistics.basic: {}", "other: {}",
                                    ".statistics.basic:\n  Covariance: {}\n  Covariance: {}"])
def test_invalid_metric_files_rejected(tmp_path, content):
    path = tmp_path / "metrics.yaml"
    path.write_text(content)
    with pytest.raises(WorkflowError):
        resolve_metrics(path)
