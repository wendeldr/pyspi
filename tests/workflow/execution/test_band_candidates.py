"""Continuous band preparation, shared views, and serial/spawn integration."""

from unittest.mock import patch

import mne
import numpy as np
import pytest
import yaml

from pyspi.data import Data
from pyspi.statistics.misc import PhaseAmplitudeCoupling, PowerEnvelopeCorrelation
from pyspi.workflow import WorkflowError, load_config, open_run, preflight, run
from pyspi.workflow._prepare import FIR, prepare_recording


@pytest.fixture
def band_config(execution_config, signal_edf, tmp_path):
    def make(*, workers=1, reference="car", **kwargs):
        names = [f"EEG A{i}" for i in range(1, 6)]
        source = signal_edf(rate=512, labels=names)
        metrics = tmp_path / "band-metrics.yaml"
        metrics.write_text(yaml.safe_dump({".statistics.misc": {
            "PowerEnvelopeCorrelation": {"configs": [{"band": [30, 70]},
                {"band": [30, 70], "orth": True, "absolute": True}]},
            "PhaseAmplitudeCoupling": {"configs": [{}]}}}))
        config = dict(source=source, metrics={"config": str(metrics)}, sampling_rate_hz=512,
                      bandpass={"low_hz": 2, "high_hz": 200}, notch={"enabled": False},
                      reference=reference, execution={"workers": workers}, output_batch_mib=.02,
                      run_name=f"bands-{workers}")
        config.update(kwargs)
        return execution_config(**config)
    return make


def test_unique_continuous_bands_match_independent_full_record_filter(band_config):
    cfg = load_config(band_config())
    with patch.object(mne.filter, "filter_data", wraps=mne.filter.filter_data) as filtering:
        prepared = prepare_recording(cfg.recordings[0], cfg)
    assert filtering.call_count == 4  # one broad pass and three distinct metric bands
    assert set(prepared.bands) == {(16., 30.), (30., 70.), (70., 150.)}
    assert len(prepared.details["continuous_bands"]) == 3
    for band, actual in prepared.bands.items():
        expected = mne.filter.filter_data(prepared.data, 512, *band, n_jobs=1, verbose="ERROR", **FIR)
        np.testing.assert_array_equal(actual, expected)
    # Explicitly check window boundaries: preparation is not filtering each slice.
    start, stop = 512, 768
    separately_filtered = mne.filter.filter_data(prepared.data[:, start:stop], 512, 30, 70,
                                                n_jobs=1, verbose="ERROR", **FIR)
    assert not np.allclose(prepared.bands[(30., 70.)][:, start:stop], separately_filtered, atol=1e-16)


def test_identical_workflow_band_is_reused(execution_config, signal_edf, tmp_path):
    metrics = tmp_path / "only-band.yaml"
    metrics.write_text(yaml.safe_dump({".statistics.misc": {
        "PowerEnvelopeCorrelation": {"configs": [{"band": [30, 70]}]}}}))
    cfg = load_config(execution_config(source=signal_edf(rate=512), sampling_rate_hz=512,
        bandpass={"low_hz": 30, "high_hz": 70}, metrics={"config": str(metrics)}, notch={"enabled": False}))
    prepared = prepare_recording(cfg.recordings[0], cfg)
    assert prepared.bands[(30., 70.)] is prepared.data
    assert prepared.details["continuous_bands"][0]["reused_workflow_bandpass"]


def test_preflight_rejects_incompatible_bands(band_config):
    with pytest.raises(WorkflowError, match="outside the continuous"):
        preflight(band_config(bandpass={"low_hz": 2, "high_hz": 100}))
    with pytest.raises(WorkflowError, match="Nyquist"):
        preflight(band_config(sampling_rate_hz=256, bandpass={"low_hz": 2, "high_hz": 100}))


def test_serial_spawned_and_manual_band_results_agree_with_contact_masks(band_config):
    names = [f"EEG A{i}" for i in range(1, 6)]
    reference = {"method": "laplacian", "shafts": {"A": names}}
    outputs = []
    for workers in (1, 2):
        cfg = load_config(band_config(workers=workers, reference=reference))
        [path] = run(cfg, max_windows=2)
        reader = open_run(path)
        assert reader.recording["channels"] == names
        values = {metric: reader.load(metric) for metric in reader.metrics}
        for metric, result in values.items():
            assert set(reader.statuses(metric)) == {"success"}
            assert np.isnan(result[[0, 4]]).all() and np.isnan(result[:, [0, 4]]).all()
            assert np.isfinite(result[1, 2]).all()
        outputs.append(values)
    for metric in outputs[0]:
        np.testing.assert_allclose(outputs[0][metric], outputs[1][metric], atol=1e-13)
    prepared = prepare_recording(cfg.recordings[0], cfg)
    metrics = [PowerEnvelopeCorrelation(band=(30, 70)),
               PowerEnvelopeCorrelation(band=(30, 70), orth=True, absolute=True), PhaseAmplitudeCoupling()]
    for window in range(2):
        start, stop = window*256, (window+1)*256
        data = Data(prepared.data[1:4, start:stop].copy(), normalise=True, detrend=True)
        data.set_filtered_bands({key: value[1:4, start:stop] for key, value in prepared.bands.items()}, sampling_rate_hz=512)
        for metric in metrics:
            np.testing.assert_allclose(outputs[0][metric.identifier][1:4, 1:4, window],
                                       metric.multivariate(data), atol=1e-13)


def test_band_calibration_tracks_views_and_prepared_memory(band_config):
    from pyspi.workflow._calibrate import calibrate
    from pyspi.workflow._estimate import estimate

    cfg = band_config()
    measured = calibrate(cfg, channels=[2, 5], windows_per_point=1)
    assert len(measured["preparation"]["details"]["continuous_bands"]) == 3
    assert all(metric["status"] == "success" for point in measured["points"] for metric in point["metrics"].values())
    estimated = estimate(cfg, measured)
    assert estimated["recordings"][0]["memory"]["prepared_bytes"] == 4 * 5 * (20*512) * 8


def test_file_backed_blocks_match_full_filter_and_honor_scratch(band_config, tmp_path):
    from pyspi.workflow._bands import prepare_band_views

    cfg = load_config(band_config(execution={"workers": 1, "scratch_directory": str(tmp_path / "scratch")}))
    prepared = prepare_recording(cfg.recordings[0], cfg)
    assert prepared.band_directory.name.startswith(str(tmp_path / "scratch"))
    assert all(isinstance(view, np.memmap) and not view.flags.writeable for view in prepared.bands.values())
    source = np.random.default_rng(4).normal(size=(5, 4096))
    views, details = prepare_band_views(source, cfg, [(30, 70)], directory=tmp_path, block_bytes=4096*8)
    expected = mne.filter.filter_data(source, 512, 30, 70, n_jobs=1, verbose="ERROR", **FIR)
    np.testing.assert_array_equal(views[(30, 70)], expected)
    assert details[0]["filter_block_channels"] == 1
    assert details[0]["storage"] == "file"
