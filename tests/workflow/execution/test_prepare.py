from fractions import Fraction

import mne
import numpy as np
import pytest

from pyspi.workflow import load_config, WorkflowError
from pyspi.workflow._prepare import FIR, harmonics, prepare_recording


@pytest.mark.parametrize("target", [128, 256, 512])
def test_continuous_sequence_matches_explicit_mne_pipeline(target, signal_edf, execution_config):
    source = signal_edf()
    path = execution_config(source=source, sampling_rate_hz=target,
                            bandpass={"low_hz": 2, "high_hz": 50},
                            recordings=[{"path": str(source), "channels": [
                                {"index": 1, "name": "B"}, {"index": 0, "name": "A"}]}])
    cfg = load_config(path)
    actual = prepare_recording(cfg.recordings[0], cfg)
    with mne.io.read_raw_edf(source, include=["EEG A1", "EEG B2"], preload=True, verbose="ERROR") as raw:
        expected = raw.get_data(picks=["EEG B2", "EEG A1"])
    expected = mne.filter.notch_filter(expected, 256, [60, 120], notch_widths=1,
                         trans_bandwidth=1, copy=False, n_jobs=1, verbose="ERROR", **FIR)
    if target != 256:
        expected = mne.filter.resample(expected, up=target, down=256, method="polyphase", n_jobs=1, verbose="ERROR")
    expected = mne.filter.filter_data(expected, target, 2, 50, copy=False, n_jobs=1, verbose="ERROR", **FIR)
    expected -= expected.mean(axis=0)
    np.testing.assert_array_equal(actual.data.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_allclose(actual.data.mean(axis=0), 0, atol=1e-20)
    assert actual.channel_names == ["B", "A"]
    assert actual.data.shape == (2, 20*target)
    assert actual.details["reference_pool"] == ["B", "A"]
    assert actual.details["bandpass"]["taps"] > 1


def test_mixed_rate_groups_preserve_native_notches(signal_edf, execution_config):
    source = signal_edf(rate=[128, 256], labels=("EEG low", "EEG high"))
    cfg = load_config(execution_config(source=source, reference="none", sampling_rate_hz=512,
                                      bandpass={"low_hz": 2, "high_hz": 100}))
    result = prepare_recording(cfg.recordings[0], cfg)
    assert result.data.shape == (2, 10240)
    groups = result.details["native_rate_groups"]
    assert [g["notch_frequencies_hz"] for g in groups] == [[60], [60, 120]]
    # Independent signal check after edge trimming: retain 10 Hz, suppress 60 Hz.
    for channel in result.data[:, 2048:-2048]:
        power = np.abs(np.fft.rfft(channel))
        freq = np.fft.rfftfreq(len(channel), 1/512)
        assert power[np.argmin(abs(freq-60))] < .002 * power[np.argmin(abs(freq-10))]


@pytest.mark.parametrize("rate,expected", [(512, [60,120,180,240]), (1024, list(range(60, 481, 60))),
                                          (2048, list(range(60, 1021, 60))), (120, [])])
def test_harmonics_respect_native_nyquist(rate, expected):
    notch = {"enabled": True, "line_frequency_hz": 60, "width_hz": 1, "transition_hz": 1}
    assert harmonics(Fraction(rate), notch) == expected
    assert harmonics(rate, {**notch, "enabled": False}) == []


@pytest.mark.parametrize("reference", ["none", {"method": "channels", "channels": ["EEG B2"]},
                                     {"method": "mapping", "mapping": {"EEG A1": ["EEG B2"]}},
                                     {"method": "bipolar", "anode": ["EEG B2"], "cathode": ["EEG A1"], "names": ["B-A"]}])
def test_reference_choices_use_retained_channel_identity(reference, execution_config):
    cfg = load_config(execution_config(reference=reference, notch={"enabled": False}))
    actual = prepare_recording(cfg.recordings[0], cfg)
    base = load_config(execution_config(source=cfg.recordings[0].path, reference="none", notch={"enabled": False}))
    unreferenced = prepare_recording(base.recordings[0], base).data
    method = reference if isinstance(reference, str) else reference["method"]
    if method == "none":
        expected = unreferenced
    elif method == "channels":
        expected = unreferenced - unreferenced[1]
    elif method == "mapping":
        expected = unreferenced.copy()
        expected[0] -= unreferenced[1]
    else:
        assert actual.channel_names == ["B-A"]
        expected = (unreferenced[1] - unreferenced[0])[None]
    np.testing.assert_array_equal(actual.data, expected)


def test_duplicate_and_empty_edf_labels_select_by_index(signal_edf, execution_config):
    source = signal_edf(labels=("EEG A", "EEG A", ""))
    cfg = load_config(execution_config(source=source, reference="none", recordings=[
        {"path": str(source), "channels": [{"index": 1, "name": "second"}, {"index": 2, "name": "empty"}]}]))
    actual = prepare_recording(cfg.recordings[0], cfg)
    assert actual.channel_names == ["second", "empty"]
    assert np.isfinite(actual.data).all()


def test_absent_reference_fails_preflight(execution_config):
    from pyspi.workflow import preflight
    path = execution_config(reference={"method": "channels", "channels": ["ECG unused"]})
    with pytest.raises(WorkflowError, match="absent from retained"):
        preflight(path)


def test_rest_with_real_mne_forward_solution(signal_edf, execution_config, tmp_path):
    names = ["EEG A", "EEG B", "EEG C"]
    source = signal_edf(labels=tuple(names))
    info = mne.create_info(names, 256, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos={
        names[0]: [0., .08, .04], names[1]: [.06, 0., .06], names[2]: [-.06, 0., .06]}, coord_frame="head"))
    src = mne.setup_volume_source_space(subject=None, pos=50., sphere=(0., 0., 0., .07), exclude=10., verbose="ERROR")
    sphere = mne.make_sphere_model(r0=(0.,0.,0.), head_radius=.09, verbose="ERROR")
    forward = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                       eeg=True, meg=False, mindist=5., n_jobs=1, verbose="ERROR")
    path = tmp_path / "test-fwd.fif"
    mne.write_forward_solution(path, forward, overwrite=True, verbose="ERROR")
    cfg = load_config(execution_config(source=source, reference={"method": "rest", "forward": str(path)}))
    actual = prepare_recording(cfg.recordings[0], cfg)
    base_cfg = load_config(execution_config(source=source, reference="none"))
    base = prepare_recording(base_cfg.recordings[0], base_cfg).data
    raw = mne.io.RawArray(base.copy(), info, verbose="ERROR")
    raw.set_eeg_reference("REST", forward=mne.read_forward_solution(path, verbose="ERROR"), verbose="ERROR")
    np.testing.assert_array_equal(actual.data, raw.get_data())
