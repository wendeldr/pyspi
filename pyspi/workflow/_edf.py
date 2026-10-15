"""EDF headers and channel identity. Never decode or resample signal payloads."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
import warnings

from ._config import WorkflowError, fields, text


class SkippedChannelsWarning(UserWarning):
    """Metadata-free selection skipped channels without the EEG-space token."""


@dataclass(frozen=True)
class Signal:
    index: int
    label: str
    samples_per_record: int
    rate: Fraction

    @property
    def annotation(self):
        return self.label in ("EDF Annotations", "BDF Annotations")

    @property
    def default_eeg(self):
        return not self.annotation and "eeg " in self.label.lower()


@dataclass(frozen=True)
class Header:
    path: Path
    records: int
    record_seconds: Fraction
    signals: tuple[Signal, ...]


def read_header(path) -> Header:
    """Read the fixed/signal headers; validate geometry using file size only.

    EDF+D cannot be treated as continuous without reading its time annotations
    and is rejected. Unknown record counts (-1) are inferred from whole records.
    Neither patient-identifying header fields nor signal samples are returned.
    """
    path = Path(path)
    try:
        with path.open("rb") as stream:
            fixed = stream.read(256)
            if len(fixed) != 256 or fixed[:8].strip() != b"0":
                raise WorkflowError(f"{path}: not a complete EDF header (BDF is unsupported)")
            header_bytes = int(fixed[184:192])
            reserved = fixed[192:236].decode("ascii").strip()
            records = int(fixed[236:244])
            duration = Decimal(fixed[244:252].decode("ascii").strip())
            count = int(fixed[252:256])
            if not duration.is_finite() or duration <= 0 or count <= 0 or records < -1:
                raise WorkflowError(f"{path}: invalid EDF duration/signal/record count")
            if reserved.startswith("EDF+D"):
                raise WorkflowError(f"{path}: discontinuous EDF+D needs a separate time-aware loader")
            minimum = 256 + 256 * count
            size = path.stat().st_size
            if header_bytes < minimum or size < header_bytes:
                raise WorkflowError(f"{path}: invalid or truncated EDF signal header")
            raw = stream.read(256 * count)
            if len(raw) != 256 * count:
                raise WorkflowError(f"{path}: truncated EDF signal header")
            labels = [raw[i * 16:(i + 1) * 16].decode("ascii").strip() for i in range(count)]
            offset = count * 216
            samples = [int(raw[offset + i * 8:offset + (i + 1) * 8]) for i in range(count)]
            if any(n <= 0 for n in samples):
                raise WorkflowError(f"{path}: EDF samples per record must be positive")
            record_bytes = 2 * sum(samples)
            payload_bytes = size - header_bytes
            actual, remainder = divmod(payload_bytes, record_bytes)
            if remainder or (records != -1 and records != actual):
                raise WorkflowError(f"{path}: EDF file size disagrees with its data-record geometry")
            seconds = Fraction(duration)
            return Header(path.resolve(), actual, seconds,
                          tuple(Signal(i, label, n, Fraction(n) / seconds)
                                for i, (label, n) in enumerate(zip(labels, samples))))
    except (OSError, UnicodeError, ValueError, InvalidOperation) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError(f"{path}: unreadable or invalid EDF header: {exc}") from exc


def select_channels(header: Header, selection=None):
    """Return ordered (Signal, output name) pairs with explicit source identity.

    A selector is an exact label, a zero-based EDF index, or a mapping containing
    exactly one of label/index and optionally name. No normalization or sorting.
    """
    selectors = selection
    if selectors is None:
        selectors = [signal.index for signal in header.signals if signal.default_eeg]
        skipped = [s.label for s in header.signals if not s.annotation and not s.default_eeg]
        if skipped:
            warnings.warn(f"{header.path}: no channel metadata; skipped channels without 'EEG ': {skipped}",
                          SkippedChannelsWarning, stacklevel=2)
    if not selectors:
        raise WorkflowError(f"{header.path}: no EEG channels selected")
    out, indices, names = [], set(), set()
    for selector in selectors:
        name = None
        if isinstance(selector, dict):
            fields(selector, {"label", "index", "name"}, "channel selector")
            if ("label" in selector) == ("index" in selector):
                raise WorkflowError("channel selector needs exactly one of label or index")
            name = text(selector["name"], "channel name") if "name" in selector else None
            selector = selector.get("label", selector.get("index"))
        if type(selector) is int:
            if not 0 <= selector < len(header.signals):
                raise WorkflowError(f"EDF channel index outside header: {selector}")
            signal = header.signals[selector]
        elif isinstance(selector, str):
            matches = [s for s in header.signals if s.label == selector]
            if len(matches) != 1:
                raise WorkflowError(f"channel label {selector!r} has {len(matches)} matches; use an explicit index for duplicates")
            signal = matches[0]
        else:
            raise WorkflowError("channel selectors must be exact labels, integer indices, or mappings")
        name = name if name is not None else signal.label
        if signal.annotation:
            raise WorkflowError("EDF annotation signals cannot be calculation channels")
        if not name:
            raise WorkflowError("an empty EDF channel label requires an explicit output name")
        if signal.index in indices or name in names:
            raise WorkflowError(f"duplicate selected source channel or output name: {name!r}")
        indices.add(signal.index)
        names.add(name)
        out.append((signal, name))
    return tuple(out)
