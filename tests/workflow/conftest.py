"""Small, valid EDF fixtures; patient recordings are never modified."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def edf(tmp_path):
    def write(labels=("EEG B2", "EEG A1"), samples=(8, 8), records=3,
              duration="1", reserved="", name="source.EDF", declared_records=None):
        path = tmp_path / name
        count = len(labels)

        def field(value, size):
            value = str(value).encode("ascii")
            assert len(value) <= size
            return value.ljust(size)

        fixed = (field("0", 8) + field("synthetic", 80) + field("synthetic", 80) +
                 field("01.01.00", 8) + field("00.00.00", 8) + field(256 + 256 * count, 8) +
                 field(reserved, 44) + field(records if declared_records is None else declared_records, 8) +
                 field(duration, 8) + field(count, 4))
        signals = b"".join(field(label, 16) for label in labels)
        for size, values in ((80, [""] * count), (8, ["uV"] * count),
                             (8, [-1] * count), (8, [1] * count),
                             (8, [-32768] * count), (8, [32767] * count),
                             (80, [""] * count), (8, samples), (32, [""] * count)):
            signals += b"".join(field(value, size) for value in values)
        assert len(fixed) == 256 and len(signals) == count * 256
        path.write_bytes(fixed + signals + bytes(records * 2 * sum(samples)))
        return path

    return write


@pytest.fixture
def metric_config(tmp_path):
    def write(classes=None, *, module=".statistics.basic", name="metrics.yaml"):
        classes = {"Covariance": {"configs": [{"estimator": "EmpiricalCovariance"}]}} if classes is None else classes
        path = tmp_path / name
        path.write_text(yaml.safe_dump({module: classes}, sort_keys=False))
        return path
    return write


@pytest.fixture
def workflow(tmp_path, metric_config):
    def write(path, *, config=None, **changes):
        cfg = {"version": 1, "recordings": [{"path": str(path)}],
               "metrics": {"config": str(metric_config())},
               "sampling_rate_hz": 8}
        if config is not None:
            cfg = config
        cfg.update(changes)
        destination = tmp_path / "workflow.yaml"
        destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return destination
    return write


@pytest.fixture(autouse=True)
def prohibit_connectivity(monkeypatch):
    from pyspi.calculator import Calculator
    from pyspi.data import Data

    def forbidden(*args, **kwargs):
        pytest.fail("preflight attempted to construct signal Data or compute connectivity")
    monkeypatch.setattr(Calculator, "compute", forbidden)
    monkeypatch.setattr(Data, "__init__", forbidden)
