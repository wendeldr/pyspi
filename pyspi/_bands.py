"""Owned, aligned band windows supplied after continuous filtering.

No filter is applied here. PEC uses Data's window preprocessing; the exploratory
PAC estimator uses the filtered samples directly, as in its source prototype.
"""

from numbers import Real

import numpy as np

from ._cache import ArrayCache, data_cache
from ._envelope import EnvelopeWorkspace


def frequency_band(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value)):
        raise ValueError("a frequency band must contain two finite positive frequencies in Hz")
    low, high = map(float, value)
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("a frequency band must satisfy 0 < low < high in Hz")
    return low, high


def band_identifier(band):
    return f"{band[0]:g}-{band[1]:g}Hz"


class BandInputs:
    def __init__(self, data, bands, sampling_rate_hz):
        if (isinstance(sampling_rate_hz, (bool, np.bool_)) or not isinstance(sampling_rate_hz, Real)
                or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0):
            raise ValueError("sampling_rate_hz must be positive and finite")
        if data.n_replications != 1:
            raise ValueError("band views require a single realization")
        if not isinstance(bands, dict):
            raise ValueError("filtered_bands must map (low_hz, high_hz) to aligned channel-by-sample arrays")
        self.source = {}
        for key, value in bands.items():
            band = frequency_band(key)
            if band[1] >= sampling_rate_hz / 2:
                raise ValueError("filtered band must lie below Nyquist")
            value = np.asarray(value)
            if value.shape != (data.n_processes, data.n_observations) or np.iscomplexobj(value):
                raise ValueError("filtered band must be real and match Data's channel/sample axes")
            self.source[band] = np.array(value, dtype=float, copy=True)
            self.source[band].flags.writeable = False
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.normalise, self.detrend = data.normalise, data.detrend
        self.epoch = data_cache(data)
        self.workspaces = {}
        self.preprocessing_reasons = {}
        self.results = ArrayCache()

    def envelope(self, band, *, preprocess):
        from .data import Data, InvalidChannelsError

        band = frequency_band(band)
        if band not in self.source:
            raise ValueError(f"missing continuously filtered band {band_identifier(band)}")
        # Without window preprocessing, both consumers can share the transform.
        preprocess = bool(preprocess and (self.normalise or self.detrend))
        key = band, preprocess
        if key not in self.workspaces:
            source = self.source[band]
            reasons = np.full(len(source), "available", dtype=object)
            if preprocess:
                transformed = np.full_like(source, np.nan)
                valid = np.flatnonzero(np.isfinite(source).all(axis=1))
                reasons[~np.isfinite(source).all(axis=1)] = "nonfinite_band_input"
                while valid.size:
                    try:
                        processed = Data(source[valid].copy(), normalise=self.normalise, detrend=self.detrend)
                        transformed[valid] = processed.to_numpy(realisation=0)
                        break
                    except InvalidChannelsError as exc:
                        reasons[valid[exc.processes]] = "unusable_band_window_variance"
                        valid = np.delete(valid, exc.processes)
                source = transformed
            self.workspaces[key] = EnvelopeWorkspace(source)
            self.preprocessing_reasons[key] = reasons
        return self.workspaces[key]

    def envelope_diagnostics(self, band, orth, log):
        work = self.envelope(band, preprocess=True)
        result = work.diagnostics(orth, log)
        key = frequency_band(band), bool(self.normalise or self.detrend)
        channel_reasons = self.preprocessing_reasons[key]
        invalid = np.flatnonzero(channel_reasons != "available")
        for name in result:
            if invalid.size:
                result[name] = result[name].astype(object)  # Do not truncate the longer reason.
            for i in invalid:
                result[name][i, :] = channel_reasons[i]
                result[name][:, i] = channel_reasons[i]
            np.fill_diagonal(result[name], "diagonal")
        return result


def band_inputs(data):
    inputs = getattr(data, "_filtered_band_inputs", None)
    if inputs is None:
        raise ValueError("this metric requires continuously filtered bands; attach aligned band windows with Data.set_filtered_bands")
    if (data_cache(data) is not inputs.epoch or data.normalise != inputs.normalise or data.detrend != inputs.detrend):
        raise ValueError("filtered bands no longer match Data; attach bands again after changing the data or preprocessing")
    return inputs


PAC_REASONS = np.array(["estimated", "nonfinite_phase_input", "unresolved_phase",
                        "nonfinite_amplitude_input", "constant_amplitude", "numerical_failure", "diagonal"])


def phase_amplitude_result(data, phase_band, amplitude_band):
    """Source-row normalized centered-amplitude/phase-vector association."""
    inputs = band_inputs(data)
    key = "pac", phase_band, amplitude_band

    def compute():
        phase = inputs.envelope(phase_band, preprocess=False)
        amplitude = inputs.envelope(amplitude_band, preprocess=False)
        z, magnitude, error = phase.analytic
        unit_amplitude, amplitude_reason = amplitude.base(False)
        resolved = (phase.input_finite & (magnitude > error).all(axis=1)
                    & np.any(phase.source != phase.source[:, :1], axis=1))
        unit_phase = np.divide(z, magnitude, out=np.zeros_like(z), where=magnitude > error)
        n = z.shape[1]
        values = abs(unit_phase @ unit_amplitude.T) / np.sqrt(n)
        reason = np.zeros(values.shape, dtype=np.uint8)
        reason[:, amplitude_reason != 0] = 4
        reason[:, ~amplitude.input_finite] = 3
        reason[~resolved, :] = 2
        reason[~phase.input_finite, :] = 1
        failure = ~np.isfinite(values) | (values > 1 + 8*n*np.finfo(float).eps)
        reason[failure & (reason == 0)] = 5
        values = np.clip(values, 0, 1)
        values[reason != 0] = np.nan
        np.fill_diagonal(values, np.nan)
        np.fill_diagonal(reason, 6)
        return values, reason

    return inputs.results.get(key, compute)
