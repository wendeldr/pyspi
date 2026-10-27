"""Caches tied to the current contents of a Data object."""

from collections import OrderedDict

import numpy as np


class ArrayCache:
    """Bound retained arrays, evicting least recently used entries first."""

    def __init__(self, max_bytes=256 * 2**20):
        self.max_bytes = max_bytes
        self.nbytes = 0
        self.entries = OrderedDict()

    @staticmethod
    def size(value):
        if isinstance(value, tuple):
            return sum(ArrayCache.size(item) for item in value)
        return value.nbytes

    def get(self, key, compute):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        value = compute()
        size = self.size(value)
        if size <= self.max_bytes:
            while self.nbytes + size > self.max_bytes:
                _, removed = self.entries.popitem(last=False)
                self.nbytes -= self.size(removed)
            self.entries[key] = value
            self.nbytes += size
        return value


class _DataCache:
    def __init__(self, source):
        # Data.to_numpy returns a writable view. A snapshot also detects edits
        # through that view or the caller's original array, not just setters.
        self.source = source.copy()
        self.families = {}


def data_cache(data):
    source = data.to_numpy()
    cache = getattr(data, "_calculation_cache", None)
    if (cache is None or cache.source.dtype != source.dtype
            or not np.array_equal(cache.source, source, equal_nan=True)):
        cache = data._calculation_cache = _DataCache(source)
    return cache


def cached_workspace(data, family, factory):
    cache = data_cache(data)
    if family not in cache.families:
        cache.families[family] = factory(cache.source)
    return cache.families[family]
