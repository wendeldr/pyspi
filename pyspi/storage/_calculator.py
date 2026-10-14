"""Explicit conversion of computed Calculator tables; no calculations run here."""

from collections.abc import Mapping

import numpy as np

from ._store import write_results


def write_calculator(path, calculator, *, metrics=None, channel_names=None,
                     recording: Mapping | None = None, **write_options):
    """Store one Calculator or a sequence of Calculators in window order.

    One Calculator produces single matrices. A sequence (even length one)
    produces ``(channels, channels, windows)`` results. Tables are aligned by
    their process labels, never sorted. ``metrics`` optionally selects/reorders
    metrics; ``channel_names`` supplies recording names in the first table's
    row order without changing numeric axes. Calculations must already have run.

    The adapter cannot distinguish a calculated NaN from an uncomputed table's
    NaN: Calculator does not expose completion status. Both are stored as values.
    Other options are passed to :func:`write_results`.
    """
    single = hasattr(calculator, "table")
    calculators = [calculator] if single else list(calculator)
    if not calculators:
        raise ValueError("at least one Calculator is required")
    first = calculators[0].table
    if not first.index.is_unique:
        raise ValueError("Calculator process labels must be unique")
    processes = list(first.index)
    if not processes:
        raise ValueError("Calculator table must contain at least one process")
    if getattr(first.columns, "nlevels", 1) != 2:
        raise ValueError("Calculator table must have (metric, process) columns")
    defaults = list(dict.fromkeys(first.columns.get_level_values(0)))
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence, not one string")
    chosen = defaults if metrics is None else list(metrics)
    if len(set(chosen)) != len(chosen):
        raise ValueError("metrics must not contain duplicates")
    results = {name: [] for name in chosen}
    labels = None
    for calc in calculators:
        table = calc.table
        if not table.index.is_unique or set(table.index) != set(processes):
            raise ValueError("all Calculator windows must contain the same unique processes")
        if getattr(table.columns, "nlevels", 1) != 2 or not table.columns.is_unique:
            raise ValueError("Calculator table must have unique (metric, process) columns")
        actual_labels = calc.get_stat_labels()
        selected_labels = {name: list(actual_labels[name]) for name in chosen}
        if labels is not None and selected_labels != labels:
            raise ValueError("metric labels must agree across Calculator windows")
        labels = selected_labels
        for name in chosen:
            block = table[name]
            if not block.columns.is_unique or set(block.columns) != set(processes):
                raise ValueError(f"{name}: Calculator row and column processes disagree")
            values = block.loc[processes, processes].to_numpy(copy=True)
            if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("Calculator values must have float32 or float64 dtype")
            if results[name] and values.dtype != results[name][0].dtype:
                raise ValueError(f"{name}: Calculator dtype must agree across windows")
            results[name].append(values)
    data = {name: arrays[0] if single else np.stack(arrays, axis=2)
            for name, arrays in results.items()}
    info = dict(recording) if recording is not None else {}
    names = processes if channel_names is None else list(channel_names)
    if isinstance(channel_names, str) or len(names) != len(processes):
        raise ValueError("channel_names must contain one name per table process")
    if "channels" in info and info["channels"] != names:
        raise ValueError("recording channels disagree with channel_names/table order")
    info["channels"] = names
    write_results(path, data, labels=labels, recording=info, **write_options)
