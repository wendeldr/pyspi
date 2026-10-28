"""Finite output overflow is recorded without interrupting calculation or resume."""
import json
import numpy as np

from pyspi.workflow import run, open_run


def test_overflow_continues_records_locations_and_survives_resume(execution_config, monkeypatch):
    from pyspi.statistics.basic import Covariance
    calls = []
    def metric(self, data):
        calls.append(1)
        return np.array([[np.nan, np.nextafter(1e308, np.inf)], [-1e308, np.nan]])
    monkeypatch.setattr(Covariance, "multivariate", metric)
    config = execution_config(output_dtype="float32", output_batch_mib=16/1024**2)
    [path] = run(config, max_windows=3)
    reader = open_run(path)
    identifier = reader.metrics[0]
    values = reader.load(identifier)
    assert len(calls) == 3 and values.dtype == np.float32
    assert np.isposinf(values[0, 1]).all() and np.isneginf(values[1, 0]).all()
    assert list(reader.statuses(identifier)) == ["success"] * 3
    events = list(reader.diagnostics(identifier))
    assert [e["window"] for e in events] == [0, 1, 2]
    for event in events:
        conversion = event["output_conversion"]
        assert conversion["metric"] == identifier
        assert conversion["count"] == 2
        assert conversion["positions"] == [[0, 1], [1, 0]]
        assert conversion["output_dtype"] == "float32"
        # JSON parsing restores the original finite float64 bits, including
        # a value whose adjacent representable neighbour differs by one ULP.
        expected = np.array([np.nextafter(1e308, np.inf), -1e308])
        restored = np.array(conversion["original_values"], dtype=conversion["source_dtype"])
        np.testing.assert_array_equal(restored.view(np.uint64), expected.view(np.uint64))
    logged = [json.loads(line) for log in sorted(path.glob("batches/*/attempts.jsonl"))
              for line in log.read_text().splitlines()]
    assert logged == events
    run(config, max_windows=3, resume=True)
    assert len(calls) == 3
    assert list(open_run(path).diagnostics(identifier)) == events
