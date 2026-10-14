"""Lossless local storage for single and windowed connectivity matrices.

Install with ``pip install 'pyspi[storage]'``. See ``docs/storage.md`` for the
public contract, ordering, metadata, partial recovery, and durability limits.
"""

from ._store import (
    Results, ShardSizeWarning, open_results, set_recording_info, verify,
    write_metric, write_results,
)
from ._integrity import (
    MetadataError, PartialReadWarning, StorageFormatError, StorageIntegrityError,
)
from ._calculator import write_calculator

__all__ = [
    "Results", "open_results", "write_metric", "write_results", "write_calculator",
    "set_recording_info", "verify", "ShardSizeWarning", "PartialReadWarning",
    "MetadataError", "StorageFormatError", "StorageIntegrityError",
]
