"""Exact sample-grid window planning, independent of signal values."""

from dataclasses import dataclass
import warnings

from ._config import WorkflowError


class IncompleteWindowWarning(UserWarning):
    """A trailing remainder is excluded from the full-window plan."""


@dataclass(frozen=True)
class WindowPlan:
    length_samples: int
    step_samples: int
    count: int

    def ranges(self):
        """Yield zero-based [start, stop) sample ranges in chronological order."""
        for index in range(self.count):
            start = index * self.step_samples
            yield start, start + self.length_samples

    def as_dict(self, *, include_ranges=True):
        result = {"length_samples": self.length_samples, "step_samples": self.step_samples,
                  "count": self.count, "stop_exclusive": True}
        if include_ranges:
            result["sample_ranges"] = list(self.ranges())
        return result


def plan_windows(samples, rate, length_seconds, step_seconds, *, recording="recording"):
    length, step = length_seconds * rate, step_seconds * rate
    if length.denominator != 1 or step.denominator != 1:
        raise WorkflowError(f"{recording}: window length/step must each span a whole number of samples at {float(rate):g} Hz")
    length, step = int(length), int(step)
    if length < 1 or step < 1:
        raise WorkflowError(f"{recording}: window length/step must each span at least one sample")
    count = max(0, (samples - length) // step + 1)
    covered = (count - 1) * step + length if count else 0
    if samples > covered:
        warnings.warn(f"{recording}: discarded incomplete final window ({samples - covered} trailing samples)",
                      IncompleteWindowWarning, stacklevel=2)
    return WindowPlan(length, step, count)
