"""Physical reference masks must survive preparation and both execution modes."""

import numpy as np
import pytest

from pyspi.workflow import load_config, open_run, preflight, run
from pyspi.workflow._prepare import prepare_recording


def test_laplacian_matches_explicit_continuous_formula(signal_edf, execution_config):
    names = [f"EEG A{i}" for i in range(1, 6)]
    source = signal_edf(labels=names)
    base_cfg = load_config(execution_config(source=source, reference="none"))
    base = prepare_recording(base_cfg.recordings[0], base_cfg)
    cfg = load_config(execution_config(source=source,
        reference={"method": "laplacian", "shafts": {"A": names}}))
    actual = prepare_recording(cfg.recordings[0], cfg)
    np.testing.assert_allclose(actual.data[1:-1], base.data[1:-1] - (base.data[:-2] + base.data[2:]) / 2,
                               atol=1e-22, rtol=1e-12)
    assert actual.channel_names == names
    assert actual.details["reference_geometry"]["available"] == [False, True, True, True, False]
    assert preflight(cfg).recordings[0]["reference_available"] == [False, True, True, True, False]


@pytest.mark.parametrize("workers", [1, 2])
def test_masked_outputs_remain_nan_without_normalization(workers, signal_edf, execution_config):
    names = [f"EEG A{i}" for i in range(1, 6)]
    source = signal_edf(labels=names)
    cfg = execution_config(source=source, reference={"method": "laplacian", "shafts": {"A": names}},
                           preprocessing={"zscore": False, "detrend": False},
                           execution={"workers": workers}, run_name=f"local-{workers}")
    [path] = run(cfg, max_windows=2)
    reader = open_run(path)
    assert reader.recording["channels"] == names
    values = reader.load(reader.metrics[0])
    assert np.isnan(values[[0, 4]]).all()
    assert np.isnan(values[:, [0, 4]]).all()
    assert np.isfinite(values[1, 2]).all()
    assert set(reader.statuses(reader.metrics[0])) == {"success"}


def test_local_reference_calibration_counts_usable_contacts(signal_edf, execution_config):
    from pyspi.workflow._calibrate import calibrate
    from pyspi.workflow._estimate import estimate

    names = [f"EEG A{i}" for i in range(1, 7)]
    source = signal_edf(labels=names)
    cfg = execution_config(source=source, reference={"method": "laplacian", "shafts": {"A": names}})
    result = calibrate(cfg, windows_per_point=1)
    assert result["ladder"] == [2, 4]
    assert result["source"]["output_channels"] == 6
    assert result["source"]["calculation_channels"] == 4
    measured = estimate(cfg, result)
    assert measured["recordings"][0]["channels"] == 6
    assert measured["recordings"][0]["calculation_channels"] == 4
    assert measured["recordings"][0]["mode"] == "measured"
    assert measured["recordings"][0]["raw_output_bytes"] == 40 * 6 * 6 * 8
    assert not any("times are extrapolated" in note for note in measured["warnings"])
