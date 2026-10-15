"""Small run index over unmodified, bounded production result stores."""

import json
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np

from ._config import WorkflowError


RUN_SCHEMA = "pyspi-eeg-run/1"


def read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"unreadable workflow metadata {path}: {exc}") from exc


def atomic_json(path, value):
    """Publish only after flushing the new JSON and its containing directory."""
    from pyspi.storage._integrity import sync_directory

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        Path(temp).unlink(missing_ok=True)


def child_path(root, relative):
    """Run indexes contain portable relative links confined to their run."""
    if not isinstance(relative, str):
        raise WorkflowError("invalid run-relative output path")
    result = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(Path(root).resolve()):
        raise WorkflowError("run output link escapes its directory")
    return result


class Journal:
    """Bounded-batch attempt log; failures are durable before continuing.

    Successes are fsynced at each window boundary. An interrupted final line is
    ignored/truncated on resume; malformed complete lines are errors.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.latest = {}
        valid_end = 0
        if self.path.exists():
            with self.path.open("rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        warnings.warn(f"{self.path}: ignoring interrupted final diagnostic line", UserWarning)
                        break
                    try:
                        event = json.loads(line)
                        key = (event["window"], event["report"]["identifier"])
                        if type(key[0]) is not int or event["report"]["status"] not in ("success", "failed"):
                            raise ValueError("invalid diagnostic identity/status")
                    except (ValueError, KeyError, TypeError) as exc:
                        raise WorkflowError(f"malformed diagnostic journal {self.path}") from exc
                    self.latest[key] = event
                    valid_end = stream.tell()
            with self.path.open("r+b") as stream:
                stream.truncate(valid_end)
        self.stream = self.path.open("a", encoding="utf-8")
        from pyspi.storage._integrity import sync_directory
        sync_directory(self.path.parent)
        sync_directory(self.path.parent.parent)

    def record(self, window, report, *, stage="calculation", window_warnings=()):
        event = {"window": window, "stage": stage, "report": report.as_dict()}
        if window_warnings:
            event["window_preprocessing_warnings"] = [asdict(w) for w in window_warnings]
        self.stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        self.latest[(window, report.identifier)] = event
        if report.status == "failed":
            self.sync()

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.sync()
        self.stream.close()


class RunResults:
    """Selective ``(N, N, time)`` access across committed window batches.

    Failed calculations are stored NaNs, with status='failed'; successfully
    calculated NaNs have status='success'. Uncommitted or unreadable data warns
    and yields unavailable NaNs, following the production reader's policy.
    ``diagnostics`` exposes committed attempts, not uncommitted journal events.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._manifest()

    def _manifest(self):
        value = read_json(self.path / "run.json")
        if value.get("schema") != RUN_SCHEMA:
            raise WorkflowError("unsupported EEG run schema")
        return value

    @property
    def metrics(self):
        return [m["identifier"] for m in self._manifest()["contract"]["metrics"]]

    @property
    def recording(self):
        data = self._manifest()
        return {"channels": data["channels"], "patient_id": data["contract"]["patient_id"],
                "recording_id": data["contract"]["recording_id"], "windows": data["windows"],
                "preparation": data.get("preparation"), "metadata": data["recording_metadata"],
                "state": data["state"]}

    def info(self, metric):
        data = self._manifest()
        spec = next((m for m in data["contract"]["metrics"] if m["identifier"] == metric), None)
        if spec is None:
            raise KeyError(metric)
        return {"n": len(data["channels"]), "windows": data["windows"]["count"],
                "dtype": data["dtypes"].get(metric, "<f8"), "single_matrix": False,
                "state": data["state"], "metric": spec}

    def load(self, metric, windows=None, *, return_availability=False):
        from pyspi.storage import Results, PartialReadWarning, open_results

        meta, manifest = self.info(metric), self._manifest()
        indices = Results._selection(meta, windows)
        shape = (meta["n"], meta["n"], len(indices))
        values = np.full(shape, np.nan, dtype=meta["dtype"])
        availability = np.zeros(shape, dtype=bool) if return_availability else None
        declared = np.zeros(len(indices), dtype=bool)
        issues = []
        for batch in manifest["batches"]:
            positions = np.flatnonzero((indices >= batch["start"]) & (indices < batch["stop"]))
            if not len(positions):
                continue
            declared[positions] = True
            if batch["state"] != "complete":
                issues.append(f"windows {batch['start']}:{batch['stop']} are unfinished")
                continue
            try:
                store = open_results(child_path(self.path, batch["path"]) / "results.zarr")
                part, good = store.load(metric, indices[positions] - batch["start"], return_availability=True)
                if part.shape != (meta["n"], meta["n"], len(positions)):
                    raise WorkflowError("batch axes disagree with run index")
                if part.dtype != values.dtype:
                    events = read_json(child_path(self.path, batch["path"]) / "reports.json")
                    if any(e["report"]["identifier"] == metric and e["report"]["status"] == "success" for e in events):
                        raise WorkflowError("batch dtype disagrees with run index")
                    # An entirely failed batch contains placeholders, not
                    # float64 observations that would constrain later precision.
                    part = np.full(part.shape, np.nan, dtype=values.dtype)
                values[:, :, positions] = part
                if availability is not None:
                    availability[:, :, positions] = good
            except (OSError, ValueError, KeyError) as exc:
                issues.append(f"windows {batch['start']}:{batch['stop']}: {exc}")
        if not declared.all():
            issues.append("selected windows have no batch declaration")
        if issues:
            warnings.warn(f"{metric}: " + "; ".join(issues[:8]), PartialReadWarning, stacklevel=2)
        if windows is not None and np.ndim(windows) == 0:
            values = values[:, :, 0]
            if availability is not None:
                availability = availability[:, :, 0]
        return (values, availability) if return_availability else values

    def diagnostics(self, metric=None):
        """Yield committed per-window reports in window/configuration order."""
        if metric is not None and metric not in self.metrics:
            raise KeyError(metric)
        for batch in self._manifest()["batches"]:
            if batch["state"] == "complete":
                for event in read_json(child_path(self.path, batch["path"]) / "reports.json"):
                    if metric is None or event["report"]["identifier"] == metric:
                        yield event

    def statuses(self, metric):
        result = np.full(self.info(metric)["windows"], "unfinished", dtype="U10")
        for event in self.diagnostics(metric):
            result[event["window"]] = event["report"]["status"]
        return result


def open_run(path):
    return RunResults(path)
