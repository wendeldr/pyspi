"""Spawned, bounded window workers; the parent owns all durable output writes."""

import atexit
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
import warnings

import numpy as np
from threadpoolctl import threadpool_limits


def peak_rss_mib():
    # Linux ru_maxrss is KiB; macOS uses bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)


def jvm_started():
    module = sys.modules.get("jpype")
    return bool(module is not None and module.isJVMStarted())


@contextmanager
def numerical_threads(count):
    """Limit loaded BLAS/OpenMP and libraries imported later; restore caller state."""
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = str(count)
    try:
        with threadpool_limits(limits=count):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass
class WindowResult:
    window: int
    valid: np.ndarray
    records: list
    warnings: tuple
    stage: str
    preprocessing_seconds: float
    calculation_seconds: float
    peak_rss_mib: float
    pid: int
    jvm_started: bool


_worker = None


def _initialize(filename, metrics, cfg, names, step, length, available=None, band_files=None):
    global _worker
    from ._run import _calculator

    directory = tempfile.TemporaryDirectory(prefix="pyspi-window-worker-")
    atexit.register(directory.cleanup)
    limits = numerical_threads(cfg.execution["threads_per_worker"])
    limits.__enter__()
    atexit.register(limits.__exit__, None, None, None)
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        calc = _calculator(metrics, cfg, Path(directory.name))
    _worker = (np.load(filename, mmap_mode="r"), calc, dict(calc.spis), names, step, length,
               np.arange(len(names)) if available is None else np.asarray(available),
               {band: np.load(path, mmap_mode="r") for band, path in (band_files or {}).items()},
               float(cfg.sampling_rate_hz))


def _compute(job):
    from pyspi.calculation import MetricReport, _capture_metric_warnings, _describe_error
    from pyspi.data import InvalidChannelsError
    from ._bands import window_band_views

    data, calc, all_spis, names, step, length, available, bands, rate = _worker
    window, active = job
    calc.spis.clear()
    calc.spis.update((identifier, all_spis[identifier]) for identifier in active)
    valid = available.copy()
    started = perf_counter()
    records = []
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        try:
            with _capture_metric_warnings() as emitted:
                while valid.size >= 2:
                    try:
                        start, stop = window*step, window*step+length
                        calc.load_dataset(data[valid, start:stop].copy(),
                            filtered_bands=window_band_views(bands, valid, start, stop) or None,
                            sampling_rate_hz=rate)
                        break
                    except InvalidChannelsError as exc:
                        excluded = [names[i] for i in valid[exc.processes]]
                        warnings.warn(f"Window {window}: skipping unusable channels {excluded}: {exc}", UserWarning)
                        valid = np.delete(valid, exc.processes)
                if valid.size < 2:
                    raise ValueError("Fewer than two usable channels remain in this window.")
        except Exception as exc:
            error = _describe_error(exc)
            records = [(MetricReport(identifier, "failed", 0., tuple(emitted), error), None)
                       for identifier in active]
            return WindowResult(window, valid, records, (), "window_preprocessing",
                                perf_counter()-started, 0., peak_rss_mib(), os.getpid(), jvm_started())
        preprocessing = perf_counter()-started
        before = perf_counter()
        calc.compute(on_metric_complete=lambda report, values: records.append((report, values)))
    return WindowResult(window, valid, records, tuple(emitted), "calculation",
                        preprocessing, perf_counter()-before, peak_rss_mib(), os.getpid(), jvm_started())


class WindowPool:
    """At most one in-flight window/result per worker, sharing read-only input."""

    def __init__(self, prepared, metrics, cfg, plan):
        from ._reference import available_indices
        from ._bands import scratch_directory
        started = perf_counter()
        self.input_directory = tempfile.TemporaryDirectory(prefix="pyspi-window-input-", dir=scratch_directory(cfg))
        directory = self.input_directory.name
        filename = Path(directory) / "prepared.npy"
        reused = {band for band, view in prepared.bands.items() if view is prepared.data}
        np.save(filename, prepared.data, allow_pickle=False)
        # Release the owned continuous array; mappings share the file's pages
        # across workers instead of copying the recording into every process.
        prepared.data = np.load(filename, mmap_mode="r")
        band_files = {}
        for i, band in enumerate(prepared.bands):
            mapped = isinstance(prepared.bands[band], np.memmap)
            path = (filename if band in reused else Path(prepared.bands[band].filename) if mapped
                    else Path(directory) / f"band-{i}.npy")
            if band not in reused and not mapped:
                np.save(path, prepared.bands[band], allow_pickle=False)
            prepared.bands[band] = np.load(path, mmap_mode="r")
            band_files[band] = str(path)
        self.workers = cfg.execution["workers"]
        self.executor = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=_initialize,
            initargs=(str(filename), metrics, cfg, prepared.channel_names,
                      plan["step_samples"], plan["length_samples"],
                      available_indices(prepared.details, len(prepared.channel_names)), band_files))
        self.setup_seconds = perf_counter()-started

    def map(self, jobs):
        jobs = iter(jobs)
        pending = set()
        for _ in range(self.workers):
            job = next(jobs, None)
            if job is not None:
                pending.add(self.executor.submit(_compute, job))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            while done:
                future = done.pop()
                result = future.result()
                del future
                yield result
                del result
                job = next(jobs, None)
                if job is not None:
                    pending.add(self.executor.submit(_compute, job))

    def close(self, *, interrupted=False):
        if interrupted:
            # Python 3.12 has no Executor.terminate_workers. Capture the owned
            # processes before shutdown, then use their public terminate/join.
            processes = list((self.executor._processes or {}).values())
            self.executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
        else:
            self.executor.shutdown(wait=True)
        self.input_directory.cleanup()
