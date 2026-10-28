"""Focused window/smoothing experiment; production settings are unchanged.

Compare recording means, not windows as independent observations. Synthetic
recordings have known null/phase-lag relationships. The optional EEG pilot
samples contacts without using EZ labels and uses the example workflow's
preprocessing. It does not evaluate EZ classification or choose optimal settings.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import detrend
from scipy.stats import spearmanr
from pyspi._spectral import SpectralConnectivity
from pyspi._spectral_compat import spectral_connectivity as sc

FS = 2048
SETTINGS = [("half_nw3", .5, 3), ("half_nw2", .5, 2),
            ("half_nw1p5", .5, 1.5), ("one_nw2", 1., 2), ("two_nw3", 2., 3)]
BANDS = [(1, 4), (4, 8), (8, 12), (13, 30), (30, 70)]
METRICS = ["msc", "wpli", "dswpli", "psi"]


@lru_cache(None)
def tapers(length, nw):
    return sc.Multitaper(np.zeros((length, 1)), sampling_frequency=FS,
                         time_halfbandwidth_product=nw).tapers


def connection(raw, nw):
    # Same window detrending/z-scoring as the example workflow, without
    # Calculator/report/storage overhead. Tapers are reusable across windows.
    x = detrend(raw, axis=1)
    x /= x.std(axis=1, ddof=1, keepdims=True)
    mt = sc.Multitaper(x.T, sampling_frequency=FS,
                      time_halfbandwidth_product=nw,
                      tapers=tapers(x.shape[1], nw))
    return SpectralConnectivity(mt.fft(), mt.frequencies, mt.time,
                                frequency_resolution=mt.frequency_resolution)


def evaluate(raw, nw, pairs):
    c = connection(raw, nw)
    out = np.empty((4, len(BANDS), len(pairs[0])))
    spectra = [abs(c.coherency()[0])**2,
               c.weighted_phase_lag_index()[0],
               c.debiased_squared_weighted_phase_lag_index()[0]]
    for b, (lo, hi) in enumerate(BANDS):
        select = (c.frequencies >= lo) & (c.frequencies <= hi)
        for m, values in enumerate(spectra):
            out[m, b] = np.mean(values[select][:, pairs[0], pairs[1]], axis=0)
        out[3, b] = c.phase_slope_index((lo, hi))[0][pairs]
    return out


def recording(raw, nw, duration, pairs):
    n = round(duration*FS)
    starts = np.arange(0, raw.shape[1]-n+1, FS//2)
    values = []
    start = time.perf_counter()
    for s in starts:
        values.append(evaluate(raw[:, s:s+n], nw, pairs))
    elapsed = time.perf_counter()-start
    values = np.asarray(values)
    stride = round(duration/.5)
    return dict(mean=np.mean(values, axis=0),
                nonoverlap_mean=np.mean(values[::stride], axis=0),
                nonfinite_fraction=float(np.mean(~np.isfinite(values))),
                windows=len(values), nonoverlap_windows=len(values[::stride]),
                seconds=elapsed)


def synthetic_raw(seed, duration=20):
    # Three pairs: independent colored noise; shared 9-11 Hz component with
    # a known phase lag; intermittent delayed coupling at constant marginal
    # variance. Color changes between records, not between parameter settings.
    rng = np.random.default_rng(seed)
    n = round(duration*FS)
    f = np.fft.rfftfreq(n, 1/FS)
    exponent = seed % 3
    color = (1+(f/15)**2)**(-exponent/4)
    def noise():
        return np.fft.irfft(np.fft.rfft(rng.normal(size=n))*color, n=n)
    x = np.stack([noise() for _ in range(6)])
    shared = np.fft.irfft(np.fft.rfft(noise())*((f >= 9) & (f <= 11)), n=n)
    x[2] += 4*shared
    x[3] += 4*np.roll(shared, 20)
    gate = (np.arange(n)/FS % 5 < .5).astype(float)
    x[5] = .9*gate*np.roll(x[4], 20) + np.sqrt(1-.81*gate)*x[5]
    return x


def summarize_samples(values):
    x = np.asarray(values)
    return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)),
                median=float(np.median(x)), minimum=float(np.min(x)),
                maximum=float(np.max(x)))


def synthetic(repetitions):
    pairs = (np.array([0, 2, 4]), np.array([1, 3, 5]))
    rows = {key: [] for key, *_ in SETTINGS}
    for rep in range(repetitions):
        raw = synthetic_raw(80260+rep)
        for key, duration, nw in SETTINGS:
            result = recording(raw, nw, duration, pairs)
            rows[key].append({k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in result.items()})
        if (rep+1) % 4 == 0:
            print(f"synthetic recordings: {rep+1}/{repetitions}", flush=True)
    summary = {}
    for key, _, _ in SETTINGS:
        means = np.array([r['mean'] for r in rows[key]])
        nonoverlap = np.array([r['nonoverlap_mean'] for r in rows[key]])
        summary[key] = {
            "null_msc_alpha": summarize_samples(means[:, 0, 2, 0]),
            "null_wpli_alpha": summarize_samples(means[:, 1, 2, 0]),
            "null_dswpli_alpha": summarize_samples(means[:, 2, 2, 0]),
            "coupled_msc_alpha": summarize_samples(means[:, 0, 2, 1]),
            "coupled_msc_theta": summarize_samples(means[:, 0, 1, 1]),
            "theta_excess_over_null": summarize_samples(means[:, 0, 1, 1]-means[:, 0, 1, 0]),
            "intermittent_msc_beta": summarize_samples(means[:, 0, 3, 2]),
            "overlap_vs_nonoverlap_abs_mean_difference": np.mean(abs(means-nonoverlap), axis=0).tolist(),
            "nonfinite_fraction": max(r['nonfinite_fraction'] for r in rows[key]),
        }
    return dict(recordings=repetitions, duration_s=20, rows=rows, summary=summary,
                interpretation="SD is across independent synthetic recording means, not windows. "
                "Narrowband population MSC is (16/17)^2 inside 9-11 Hz, zero outside; "
                "band-grid endpoints differ between durations. Intermittent pair has 10% coupling duty cycle. "
                "Phase lag of shared activity does not establish physical causality.")


def eeg_choices(root):
    entries = []
    for store in sorted(root.glob('*.zarr')):
        a = json.loads((store/'patient_info/zarr.json').read_text())['attributes']
        entries.append((store, a['sampling_rate_hz'], a['node_count']))
    chosen = []
    for rate in sorted({e[1] for e in entries}):
        group = sorted([e for e in entries if e[1] == rate], key=lambda e: (e[2], e[0].name))
        indices = [0, len(group)//2, len(group)-1] if rate == 2048 else [len(group)//2]
        chosen.extend(group[i][0] for i in sorted(set(indices)))
    return chosen


def read_eeg(store, contacts=12):
    import mne
    import zarr
    from pyspi.workflow._edf import read_header
    from pyspi.workflow._metadata import _contact_key
    from pyspi.workflow._prepare import FIR, harmonics
    g = zarr.open_group(str(store), mode='r')
    names = g['patient_info/electrode'][:].tolist()
    pid = store.stem
    edfs = list((store.parent.parent/'baseline_edfs').glob(f'{pid}_*.EDF'))
    if len(edfs) != 1:
        raise ValueError(f'{pid}: expected exactly one EDF')
    edf = edfs[0]
    header = read_header(edf)
    lookup = {}
    for signal in header.signals:
        if not signal.annotation:
            lookup.setdefault(_contact_key(signal.label), []).append(signal)
    selected = []
    for name in names:
        matches = lookup.get(_contact_key(name), [])
        if len(matches) != 1:
            raise ValueError(f'{pid}: ambiguous/missing channel mapping')
        selected.append(matches[0])
    if len({s.index for s in selected}) != len(selected):
        raise ValueError(f'{pid}: duplicate retained electrode mapping')
    rates = {s.rate for s in selected}
    if len(rates) != 1:
        raise ValueError(f'{pid}: mixed rates require the full workflow reader')
    rate = float(rates.pop())
    rng = np.random.default_rng(82500+int(pid))
    indices = np.sort(rng.choice(len(names), min(contacts, len(names)), replace=False))
    # Preserve the complete retained-contact CAR pool while only retaining a
    # small label-blind contact sample. Linear identical filters/resampling
    # commute with the channel mean, so filter the selected channels and mean.
    with mne.io.read_raw_edf(str(edf), preload=False, infer_types=False,
            stim_channel=None, exclude_after_unique=True, verbose='ERROR') as probe:
        ordinary = [s for s in header.signals if not s.annotation]
        mapping = {s.index: name for s, name in zip(ordinary, probe.ch_names)}
    include = [mapping[s.index] for s in selected]
    with mne.io.read_raw_edf(str(edf), include=include, preload=False, infer_types=False,
            stim_channel=None, exclude_after_unique=True, verbose='ERROR') as raw:
        if raw.info['sfreq'] != rate:
            raise ValueError('Unexpected resampling by EDF reader')
        values = np.empty((len(indices)+1, raw.n_times))
        for start in range(0, raw.n_times, round(10*rate)):
            stop = min(start+round(10*rate), raw.n_times)
            block = raw.get_data(picks=include, start=start, stop=stop)
            values[:-1, start:stop] = block[indices]
            values[-1, start:stop] = block.mean(axis=0)
    notch = dict(enabled=True, line_frequency_hz=60, width_hz=1, transition_hz=1)
    freqs = harmonics(rate, notch)
    values = mne.filter.notch_filter(values, rate, freqs, notch_widths=1,
                                   trans_bandwidth=1, n_jobs=1, copy=False, verbose='ERROR', **FIR)
    if rate != FS:
        values = mne.filter.resample(values, up=FS, down=rate, method='polyphase',
                                    n_jobs=1, verbose='ERROR')
    values = mne.filter.filter_data(values, FS, .5, 500, copy=False,
                                   n_jobs=1, verbose='ERROR', **FIR)
    result = values[:-1]-values[-1]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite EEG input')
    return result, dict(patient_id=pid, native_rate_hz=rate,
        retained_contacts=len(names), sampled_contact_indices=indices.tolist(),
        duration_s=result.shape[1]/FS, edf_size_bytes=edf.stat().st_size,
        electrode_order_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest())


def real_pilot(root):
    reports = []
    for store in eeg_choices(root):
        print(f"EEG preparing {store.stem}", flush=True)
        started = time.perf_counter()
        raw, metadata = read_eeg(store)
        metadata['preparation_seconds'] = time.perf_counter()-started
        pairs = np.triu_indices(raw.shape[0], 1)
        rows = {}
        for key, duration, nw in SETTINGS:
            print(f"EEG {store.stem}: {key}", flush=True)
            r = recording(raw, nw, duration, pairs)
            rows[key] = r
        baseline = rows[SETTINGS[0][0]]['mean']
        for key, r in rows.items():
            difference = r['mean']-baseline
            r['comparison_to_baseline'] = []
            for m, metric in enumerate(METRICS):
                for b, band in enumerate(BANDS):
                    ref, actual = baseline[m, b], r['mean'][m, b]
                    finite = np.isfinite(ref) & np.isfinite(actual)
                    rho = float(spearmanr(ref[finite], actual[finite]).statistic) if finite.sum()>2 else None
                    r['comparison_to_baseline'].append(dict(metric=metric, band=band,
                        mean_absolute_difference=float(np.mean(abs(difference[m, b][finite]))),
                        median_baseline=float(np.median(ref[finite])),
                        median_candidate=float(np.median(actual[finite])),
                        edge_rank_correlation=rho,
                        overlap_vs_nonoverlap_mean_absolute_difference=float(
                            np.mean(abs(r['mean'][m, b]-r['nonoverlap_mean'][m, b])))))
            for k in ('mean', 'nonoverlap_mean'):
                r[k] = r[k].tolist()
        reports.append(dict(metadata=metadata, settings=rows))
    return dict(patients=reports, selection="One middle contact-count record per uncommon native rate; "
        "minimum/middle/maximum contact count at 2048 Hz. Twelve random retained contacts per patient, "
        "chosen without accessing EZ targets. Full available recording, all complete windows.",
        preprocessing="Example workflow assumptions: native 60-Hz harmonic notches, polyphase to 2048 Hz, "
        "0.5-500 Hz FIR, CAR across all retained electrodes; linear detrend/z-score per window. "
        "User's historical preprocessing not confirmed. This is not an EZ discrimination test.")


def timings():
    raw = np.random.default_rng(823).normal(size=(32, 2*FS))
    pairs = np.triu_indices(32, 1)
    walls = {key: [] for key, *_ in SETTINGS}
    for key, duration, nw in SETTINGS:
        evaluate(raw[:, :round(duration*FS)], nw, pairs)
    for rep in range(3):
        order = SETTINGS if rep % 2 == 0 else SETTINGS[::-1]
        for key, duration, nw in order:
            started = time.perf_counter()
            evaluate(raw[:, :round(duration*FS)], nw, pairs)
            walls[key].append(time.perf_counter()-started)
    return {key: dict(walls_s=x, median_s=float(np.median(x))) for key, x in walls.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-recordings', type=int, default=16)
    parser.add_argument('--eeg-root', type=Path)
    parser.add_argument('--resume-eeg', action='store_true',
                        help='Read the saved synthetic/timing receipt; run only the EEG stage.')
    parser.add_argument('--output', type=Path, default=ROOT/'audit/briefing/spectral_window_smoothing.json')
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    report = dict(settings=[dict(name=k, duration_s=d, nw=nw,
                                actual_tapers=tapers(round(d*FS), nw).shape[-1],
                                full_smoothing_hz=2*nw/d) for k, d, nw in SETTINGS],
                  bands=BANDS, metrics=METRICS, sampling_rate_hz=FS, step_s=.5,
                  cpu=cpu, numerical_threads=1,
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ['pyspi/_spectral.py', str(Path(__file__).relative_to(ROOT))]},
                  limitations=["Small exploratory experiment; no optimal setting or cross-patient EZ claim.",
                    "Benchmark is 32 channels/four estimators, not the full retained185 workload.",
                    "Directed factorization remains under review and is not used to choose smoothing here.",
                    "No production code, default settings, patient data, or regression goldens are changed."])
    started = time.perf_counter()
    if args.resume_eeg:
        if args.eeg_root is None:
            parser.error('--resume-eeg requires --eeg-root')
        current_hashes = report['source_sha256']
        report = json.loads(args.output.read_text())
        report['eeg_stage_source_sha256'] = current_hashes
    else:
        report['synthetic'] = synthetic(args.synthetic_recordings)
        report['timing_32_channels'] = timings()
        report['synthetic_and_timing_seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('Synthetic and timing receipt saved', flush=True)
    if args.eeg_root:
        report['eeg'] = real_pilot(args.eeg_root)
    report['elapsed_this_invocation_seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f"Finished in {report['elapsed_this_invocation_seconds']:.1f} s: {args.output}", flush=True)


if __name__ == '__main__':
    main()
