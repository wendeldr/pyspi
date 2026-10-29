"""Bounded downstream derivations from one saved run; no EEG recalculation.

The iterator leaves temporal averaging and persistence to the caller. Never
concatenate batches from different provenance identities as one time series.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._config import WorkflowError
from ._output import open_run


@dataclass
class DerivedBatch:
    window_indices: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    provenance: dict


def iter_derived_windows(run, metric, *, transform="identity", absolute_change=False,
                         windows=None, batch_windows=16):
    """Yield (N,N,time) float64 batches with explicit pairwise validity.

    ``pearson``/``absolute_pearson`` require stored cosine distance and recorded
    z-scoring (which guarantees centered, nondegenerate accepted channels).
    ``absolute_change`` follows the selected transform and compares consecutive
    original window indices only. The first requested window and gaps are NaN;
    failed/unfinished, nonfinite and unavailable values cannot bridge a gap.
    An identity transform permits changes of any saved metric.
    """
    if transform not in ("identity", "pearson", "absolute_pearson"):
        raise WorkflowError("transform must be identity, pearson or absolute_pearson")
    if type(absolute_change) is not bool:
        raise WorkflowError("absolute_change must be a boolean")
    if type(batch_windows) is not int or batch_windows < 1:
        raise WorkflowError("batch_windows must be a positive integer")
    reader = open_run(run) if isinstance(run, (str, Path)) else run
    info, recording = reader.info(metric), reader.recording
    if transform != "identity":
        spec = info["metric"]
        if (spec["module"] not in (".statistics.distance", "pyspi.statistics.distance")
                or spec["class_name"] != "PairwiseDistance"
                or spec["parameters"].get("metric") != "cosine"):
            raise WorkflowError("Pearson derivation requires saved PairwiseDistance(metric='cosine')")
        if not recording["preprocessing"]["zscore"]:
            raise WorkflowError("Pearson derivation requires recorded z-scoring to verify centering and nonzero variance")
    count = info["windows"]
    if windows is None:
        indices = np.arange(count)
    else:
        indices = np.asarray(windows)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= count)
                or np.any(np.diff(indices.astype(np.int64)) <= 0)):
            raise WorkflowError("windows must be strictly increasing integer indices within this run")
    statuses = np.asarray(reader.statuses(metric))
    provenance = {"run": str(Path(reader.path).resolve()), "recording": recording,
                  "input_metric": info["metric"], "transform": transform,
                  "absolute_change": absolute_change, "temporal_summary": None}
    previous, previous_valid, previous_index = None, None, None
    for offset in range(0, len(indices), batch_windows):
        selected = indices[offset:offset+batch_windows]
        raw, available = reader.load(metric, selected, return_availability=True)
        values = np.array(raw, dtype=np.float64, copy=True)
        valid = np.asarray(available, dtype=bool) & np.isfinite(values)
        valid &= (statuses[selected] == "success")[None, None, :]
        diagonal = np.arange(info["n"])
        valid[diagonal, diagonal, :] = False
        if transform != "identity":
            values = 1.-values
            if transform == "absolute_pearson":
                np.abs(values, out=values)
        values[~valid] = np.nan
        if absolute_change:
            changed = np.full_like(values, np.nan)
            changed_valid = np.zeros_like(valid)
            for slot, index in enumerate(selected):
                if previous_index is not None and index == previous_index+1:
                    good = valid[:, :, slot] & previous_valid
                    with np.errstate(over="ignore", invalid="ignore"):
                        difference = abs(values[:, :, slot]-previous)
                    good &= np.isfinite(difference)
                    changed[:, :, slot] = np.where(good, difference, np.nan)
                    changed_valid[:, :, slot] = good
                previous = values[:, :, slot].copy()
                previous_valid = valid[:, :, slot].copy()
                previous_index = int(index)
            values, valid = changed, changed_valid
        yield DerivedBatch(selected.copy(), values, valid, provenance)
