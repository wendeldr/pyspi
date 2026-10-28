"""Conservative resource planning from measured process peaks, without running data."""

import math
import os
from pathlib import Path
import threading


def available_memory_mib():
    try:
        entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(entries["MemAvailable"].split()[0]) / 1024
    except (OSError, KeyError, ValueError):
        return None


def estimate_workers(*, memory_mib, worker_peak_mib, parent_peak_mib, cpus=None,
                     threads_per_worker=1, reserve_fraction=.2):
    """An advisory limit; measured peaks must match the intended workload/cache.

    Summing worker RSS overcounts shared pages conservatively. This is not an
    operating-system memory limit and does not promise linear CPU scaling.
    """
    for name, value in (("memory_mib", memory_mib), ("worker_peak_mib", worker_peak_mib)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(parent_peak_mib) or parent_peak_mib < 0:
        raise ValueError("parent_peak_mib must be finite and nonnegative")
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    cpus = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1) if cpus is None else cpus
    for name, value in (("cpus", cpus), ("threads_per_worker", threads_per_worker)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    usable = max(0., memory_mib * (1-reserve_fraction) - parent_peak_mib)
    memory_workers = math.floor(usable / worker_peak_mib)
    cpu_workers = cpus // threads_per_worker
    workers = min(memory_workers, cpu_workers)
    return {"workers_upper_bound": workers, "memory_limited_workers": memory_workers,
            "cpu_limited_workers": cpu_workers, "memory_mib": memory_mib,
            "worker_peak_mib": worker_peak_mib, "parent_peak_mib": parent_peak_mib,
            "cpus": cpus, "threads_per_worker": threads_per_worker,
            "reserve_fraction": reserve_fraction,
            "estimated_total_mib": parent_peak_mib + workers*worker_peak_mib,
            "limits": "Advisory upper bound from supplied measured RSS, not a memory guarantee or throughput prediction. Include imports/JVM, caches, inputs and output buffers in measurements; benchmark increasing worker counts."}


class ProcessTreeMemory:
    """Sample Linux process-tree RSS/PSS, including spawned workers and helpers.

    PSS apportions shared pages instead of counting them once per process.
    Samples may miss short-lived peaks; ru_maxrss is recorded separately by the
    runner. Unsupported/inaccessible counters remain unavailable, never zero.
    """

    def __init__(self, interval=.05):
        self.pid = os.getpid()
        self.interval = interval
        self.stop = threading.Event()
        self.samples = 0
        self.peak_rss_mib = None
        self.peak_pss_mib = None
        self.peak_processes = 0
        self.minimum_available_mib = None
        self.errors = 0

    def _sample(self):
        pending, seen = [self.pid], set()
        rss, pss = 0., 0.
        complete = True
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                pending.extend(int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
                values = {}
                for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                    key, _, value = line.partition(":")
                    if key in ("Rss", "Pss"):
                        values[key] = int(value.split()[0]) / 1024
                rss += values["Rss"]
                pss += values["Pss"]
            except (OSError, ValueError, KeyError):
                complete = False
                self.errors += 1
        if complete:
            self.samples += 1
            self.peak_rss_mib = max(self.peak_rss_mib or 0, rss)
            self.peak_pss_mib = max(self.peak_pss_mib or 0, pss)
            self.peak_processes = max(self.peak_processes, len(seen))
        available = available_memory_mib()
        if available is not None:
            self.minimum_available_mib = min(self.minimum_available_mib if self.minimum_available_mib is not None else available, available)

    def _watch(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()

    def as_dict(self):
        return {"peak_rss_mib": self.peak_rss_mib, "peak_pss_mib": self.peak_pss_mib,
                "peak_processes": self.peak_processes, "samples": self.samples,
                "interval_seconds": self.interval, "counter_read_errors": self.errors,
                "minimum_system_available_mib": self.minimum_available_mib,
                "limits": "Linux sampled process-tree counters; short peaks may be missed. PSS accounts for shared pages; RSS double-counts them. System availability includes other jobs."}
