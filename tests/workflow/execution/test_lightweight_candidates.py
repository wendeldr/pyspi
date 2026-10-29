"""New candidates use the existing precision, process and resume contracts."""
import numpy as np
import yaml

from pyspi.workflow import load_config, open_run, run


def test_serial_spawn_and_resume(execution_config, signal_edf, tmp_path):
    metrics = tmp_path / 'lightweight.yaml'
    metrics.write_text(yaml.safe_dump({
        '.statistics.basic': {
            'FixedLagCorrelation': {'configs': [{}]},
            'ThresholdEventExcess': {'configs': [{}]},
            'LogScaleRatio': {'configs': [{}]},
            'PartialCorrelation': {'configs': [{}]},
            'Precision': {'configs': [{'estimator': 'LedoitWolf'}]},
        },
        '.statistics.distance': {
            'HistogramRelativeEntropy': {'configs': [{}]},
            'HistogramJensenShannon': {'configs': [{}]},
        },
    }))
    source = signal_edf(labels=('EEG A1', 'EEG A2', 'EEG A3', 'EEG A4'))
    results = []
    for workers in (1, 2):
        cfg = load_config(execution_config(source=source, metrics={'config': str(metrics)},
            execution={'workers': workers}, run_name=f'lightweight-{workers}',
            preprocessing={'detrend': True, 'zscore': False}, output_dtype='float32'))
        [path] = run(cfg, max_windows=2)
        reader = open_run(path)
        result = {name: reader.load(name) for name in reader.metrics}
        assert len(result) == 7
        assert all(set(reader.statuses(name)) == {'success'} for name in result)
        assert all(value.dtype == np.float32 for value in result.values())
        assert all(np.isfinite(value[0, 1]).all() for value in result.values())
        [resumed] = run(cfg, max_windows=2, resume=True)
        assert resumed == path
        for name, value in result.items():
            np.testing.assert_array_equal(open_run(path).load(name), value)
        results.append(result)
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name])
