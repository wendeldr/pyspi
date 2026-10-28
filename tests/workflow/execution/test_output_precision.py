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


def test_spawned_worker_overflow_records_original_values(execution_config, metric_config):
    # EDF fixture units are microvolts. Unnormalized precision has large finite
    # values that overflow float16, exercising the real worker-to-parent path.
    metrics=metric_config({'Precision':{'configs':[{'estimator':'OAS'}]}})
    cfg=execution_config(metrics={'config':str(metrics)},output_dtype='float16',
                         preprocessing={'detrend':True,'zscore':False},
                         execution={'workers':2,'threads_per_worker':1})
    [path]=run(cfg,max_windows=2)
    reader=open_run(path);name=reader.metrics[0]
    events=list(reader.diagnostics(name))
    assert len(events)==2
    assert list(reader.statuses(name))==['success','success']
    for event in events:
        conversion=event['output_conversion']
        assert conversion['count']>0
        assert len(conversion['positions'])==len(conversion['original_values'])==conversion['count']
        assert np.isfinite(conversion['original_values']).all()
        for (row,col),value in zip(conversion['positions'],conversion['original_values']):
            saved=reader.load(name,windows=event['window'])[row,col]
            assert np.isinf(saved) and np.signbit(saved)==np.signbit(value)


def test_mismatched_stored_dtype_is_unavailable_and_cannot_resume(execution_config):
    import pytest
    from seeg_dataset import storage
    from pyspi.workflow import WorkflowError
    cfg=execution_config(output_dtype='float32')
    [path]=run(cfg,max_windows=1)
    reader=open_run(path);name=reader.metrics[0]
    manifest=json.loads((path/'run.json').read_text())
    batch=path/manifest['batches'][0]['path']/'results.zarr'
    values=storage.open_results(batch).load(name).astype(np.float64)
    storage.write_metric(batch,name,values,overwrite=True)
    with pytest.warns(storage.PartialReadWarning,match='dtype disagrees'):
        _,readable=reader.load(name,return_availability=True)
    assert not readable.any()
    with pytest.raises(WorkflowError,match='axes/dtype'):
        run(cfg,resume=True,max_windows=1)
