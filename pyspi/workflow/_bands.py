"""Discover and prepare shared continuous band views for configured metrics."""

from time import perf_counter

from ._config import WorkflowError


def scratch_directory(cfg):
    """Prefer explicit scratch, otherwise the output filesystem, then OS temp."""
    from pathlib import Path
    path = cfg.execution.get("scratch_directory")
    if path is None and cfg.output_directory is not None:
        path = cfg.output_directory / ".pyspi-scratch"
    if path is not None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def required_bands(metrics, cfg):
    bands = tuple(sorted({tuple(band) for metric in metrics for band in metric.continuous_bands}))
    for low, high in bands:
        if not cfg.bandpass[0] <= low < high <= cfg.bandpass[1]:
            raise WorkflowError(f"requested band {low:g}–{high:g} Hz lies outside the continuous workflow bandpass")
    return bands


def prepare_band_views(data, cfg, bands, *, directory=None, block_bytes=16*2**20):
    """Filter each unique band once; optionally retain disk-backed continuous views.

    Channel blocks limit anonymous temporaries without creating time boundaries.
    Full-recording band arrays stay on disk and workers map the same files.
    """
    from pathlib import Path
    import mmap
    import mne
    import numpy as np
    from ._prepare import FIR

    views, details = {}, []
    rate = float(cfg.sampling_rate_hz)
    for number, (low, high) in enumerate(bands):
        started = perf_counter()
        reused = (low, high) == cfg.bandpass
        taps = mne.filter.create_filter(None, rate, low, high, verbose="ERROR", **FIR)
        block = max(1, block_bytes // (data.shape[1]*8))
        if reused:
            views[(low, high)] = data
        elif directory is None:
            views[(low, high)] = mne.filter.filter_data(
                data, rate, low, high, copy=True, n_jobs=1, verbose="ERROR", **FIR)
        else:
            filename = Path(directory) / f"band-{number}.npy"
            out = np.lib.format.open_memmap(filename, mode="w+", dtype=np.float64, shape=data.shape)
            for start in range(0, len(data), block):
                out[start:start+block] = mne.filter.filter_data(data[start:start+block], rate, low, high,
                    copy=True, n_jobs=1, verbose="ERROR", **FIR)
                out.flush()
                if hasattr(out._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    out._mmap.madvise(mmap.MADV_DONTNEED)
            del out
            views[(low, high)] = np.load(filename, mmap_mode="r")
        details.append({"low_hz": low, "high_hz": high, **FIR, "taps": len(taps),
                        "reused_workflow_bandpass": reused,
                        "storage": "shared_base" if reused else "file" if directory is not None else "memory",
                        "elapsed_seconds": perf_counter() - started,
                        "filter_block_channels": min(len(data), block) if directory is not None else len(data),
                        "lower_transition_hz": min(max(low * .25, 2), low),
                        "upper_transition_hz": min(max(high * .25, 2), rate / 2 - high),
                        "analytic_transform": "within each window after continuous filtering"})
    return views, details


def window_band_views(bands, indices, start, stop):
    return {band: source[indices, start:stop] for band, source in bands.items()}
