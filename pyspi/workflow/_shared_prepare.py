"""Bounded, invocation-local disk cache for compatible continuous preparation.

Only time-independent linear references commute with the continuous band FIRs.
Other references use the existing preparation path. Window normalization never
enters this cache: it is still performed separately by each homogeneous job.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import mmap
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from ._bands import scratch_directory
from ._prepare import PreparedRecording, prepare_recording
from ._reference import apply_shaft_reference, reference_plan, reference_details


SHAREABLE = {"none", "car", "adaptive_local", "shaft_average", "laplacian"}


def preparation_key(recording, cfg):
    from ._run import _source_identity
    identity = dict(source=_source_identity(recording.path), channels=recording.channels,
                    patient=recording.patient_id, recording=recording.recording_id,
                    sampling_rate_hz=str(cfg.sampling_rate_hz), bandpass=cfg.bandpass, notch=cfg.notch)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _disk_array(path, values):
    np.save(path, values, allow_pickle=False)
    return np.load(path, mmap_mode="r")


def _release_pages(values):
    if isinstance(values, np.memmap) and hasattr(values._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        values._mmap.madvise(mmap.MADV_DONTNEED)


def _reference_array(source, names, reference, directory, name, block_bytes=16*2**20):
    """Time blocking is exact for fixed instantaneous reference weights."""
    path = Path(directory)/name
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=source.shape)
    block = max(1, block_bytes//(len(source)*8))
    plan = reference_plan(names, reference)
    for start in range(0, source.shape[1], block):
        values = source[:, start:start+block]
        if reference['method'] == 'car':
            out[:, start:start+block] = values-values.mean(axis=0, keepdims=True)
        else:
            out[:, start:start+block] = apply_shaft_reference(values, names, reference, plan)
        out.flush()
        _release_pages(out)
    del out
    _release_pages(source)
    return np.load(path, mmap_mode="r")


class SharedPreparation:
    def __init__(self, *, max_bytes, band_unions):
        self.max_bytes = max_bytes
        self.band_unions = band_unions
        self.entries = OrderedDict()
        self.bytes = 0
        self.stats = dict(base_misses=0, base_hits=0, reference_misses=0, reference_hits=0,
                          evictions=0, peak_retained_bytes=0, cache_limit_bytes=max_bytes)

    @staticmethod
    def size(prepared):
        arrays = {str(value.filename): value for value in (prepared.data, *prepared.bands.values())}
        return sum(Path(name).stat().st_size for name in arrays)

    def _get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def _put(self, key, prepared):
        size = self.size(prepared)
        if size > self.max_bytes:
            return
        while self.bytes+size > self.max_bytes:
            _, (_, previous_size) = self.entries.popitem(last=False)
            self.bytes -= previous_size
            self.stats['evictions'] += 1
        self.entries[key] = prepared, size
        self.bytes += size
        self.stats['peak_retained_bytes'] = max(self.stats['peak_retained_bytes'], self.bytes)

    def close(self):
        self.entries.clear()
        self.bytes = 0

    def __call__(self, recording, cfg, *, bands):
        started = perf_counter()
        if cfg.reference['method'] not in SHAREABLE or self.max_bytes == 0:
            return prepare_recording(recording, cfg, bands=bands)
        base_key = preparation_key(recording, cfg)
        union = self.band_unions.get(base_key, tuple(bands))
        reference_key = ('reference', base_key, json.dumps(cfg.reference, sort_keys=True), tuple(union))
        cached = self._get(reference_key)
        if cached is not None:
            self.stats['reference_hits'] += 1
        else:
            self.stats['reference_misses'] += 1
            base = self._get(('base', base_key, tuple(union)))
            if base is None:
                self.stats['base_misses'] += 1
                base = prepare_recording(recording, replace(cfg, reference={'method': 'none'}), bands=union)
                if base.band_directory is None:
                    base.band_directory = TemporaryDirectory(prefix='pyspi-base-', dir=scratch_directory(cfg))
                original = base.data
                base.data = _disk_array(Path(base.band_directory.name)/'base.npy', original)
                for band, view in base.bands.items():
                    if view is original:
                        base.bands[band] = base.data
                del original
                _release_pages(base.data)
                self._put(('base', base_key, tuple(union)), base)
            else:
                self.stats['base_hits'] += 1
            if cfg.reference['method'] == 'none':
                cached = base
                # Same files must not count twice in the retained cache budget.
            else:
                directory = TemporaryDirectory(prefix='pyspi-reference-', dir=scratch_directory(cfg))
                data = _reference_array(base.data, base.channel_names, cfg.reference, directory.name, 'base.npy')
                views = {band: data if values is base.data else _reference_array(
                    values, base.channel_names, cfg.reference, directory.name, f'band-{i}.npy')
                    for i, (band, values) in enumerate(base.bands.items())}
                details = deepcopy(base.details)
                details['reference'] = deepcopy(cfg.reference)
                details['reference_pool'] = base.channel_names if cfg.reference['method'] == 'car' else None
                details['reference_geometry'] = reference_details(base.channel_names, reference_plan(base.channel_names, cfg.reference))
                details['shared_band_operation_order'] = 'continuous band FIR then fixed instantaneous reference (commuting linear operations)'
                cached = PreparedRecording(data, list(base.channel_names), details, perf_counter()-started, views, directory)
                self._put(reference_key, cached)
        # The worker pool may replace mapping objects. Never expose the cache's
        # mutable wrapper/dictionaries; retain the directory owner until done.
        details = deepcopy(cached.details)
        details['continuous_bands'] = [b for b in details.get('continuous_bands', [])
                                       if (b['low_hz'], b['high_hz']) in bands]
        details['preparation_cache'] = dict(self.stats)
        return PreparedRecording(cached.data, list(cached.channel_names), details, perf_counter()-started,
                                 {band: cached.bands[band] for band in bands}, cached.band_directory)
