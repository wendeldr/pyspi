"""Small, serializable diagnostics for completed Calculator metric attempts."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import traceback
from typing import Literal
import warnings


@dataclass(frozen=True, slots=True)
class MetricWarning:
    """A warning emitted under the caller's active warning filters."""

    category: str
    message: str
    filename: str
    lineno: int


@dataclass(frozen=True, slots=True)
class MetricError:
    """Exception details without retaining the exception or its local arrays."""

    type: str
    message: str
    traceback: str


@dataclass(frozen=True, slots=True)
class MetricReport:
    """One finished attempt; success includes successfully returned NaNs/infs.

    Values live in Calculator.table and the optional completion callback, not
    in this record. Completion here describes calculation, not durable storage.
    """

    identifier: str
    status: Literal["success", "failed"]
    elapsed_seconds: float
    warnings: tuple[MetricWarning, ...]
    error: MetricError | None

    def as_dict(self):
        """Return independent JSON-compatible diagnostic fields."""
        return asdict(self)


def _describe_error(error):
    cls = type(error)
    return MetricError(
        type=f"{cls.__module__}.{cls.__qualname__}",
        message=str(error),
        traceback="".join(traceback.format_exception(error)),
    )


@contextmanager
def _capture_metric_warnings():
    """Observe emitted warnings and forward them to the existing handler.

    Do not change filters or enter catch_warnings: that invalidates warning
    registries and changes suppression of repeated warnings. Python 3.12's
    showwarning hook is process-global; concurrent threaded computes are not
    supported. The subsequent runner's process isolation belongs elsewhere.
    """
    captured = []
    original = warnings.showwarning

    def observe(message, category, filename, lineno, file=None, line=None):
        captured.append(MetricWarning(
            category=f"{category.__module__}.{category.__qualname__}",
            message=str(message), filename=filename, lineno=lineno,
        ))
        original(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = observe
    try:
        yield captured
    finally:
        warnings.showwarning = original
