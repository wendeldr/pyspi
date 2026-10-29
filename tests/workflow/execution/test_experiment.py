"""Grouped preparation equivalence, bounded reuse, contracts and interruption recovery."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from pyspi.workflow import load_config, open_run, preflight, run
from pyspi.workflow._prepare import prepare_recording
from pyspi.workflow._shared_prepare import SharedPreparation, preparation_key
from pyspi.workflow.experiment import load_experiment, preflight_experiment, run_experiment
from pyspi.workflow._config import WorkflowError


@pytest.fixture
def experiment_file(execution_config, signal_edf, tmp_path):
    def make(*, workers=1, cache=8):
        source = signal_edf(rate=512, records=3, labels=('EEG A1', 'EEG A2', 'EEG A3', 'EEG A4'))
        metrics = tmp_path/'group-metrics.yaml'
        metrics.write_text(yaml.safe_dump({'.statistics.basic': {'LogScaleRatio': {'configs': [{}]}},
            '.statistics.misc': {'PowerEnvelopeCorrelation': {'configs': [{'band': [30, 70]}]}}}))
        standalone = execution_config(source=source, sampling_rate_hz=512,
            bandpass={'low_hz': 2, 'high_hz': 200}, notch={'enabled': False},
            metrics={'config': str(metrics)}, output_batch_mib=1,
            execution={'workers': workers, 'scratch_directory': str(tmp_path/'scratch')},
            output_dtype='float32')
        document = yaml.safe_load(standalone.read_text())
        document['groups'] = {'normalized': {}, 'amplitude': {'preprocessing': {'zscore': False}},
                              'shaft': {'reference': {'method': 'shaft_average', 'shafts': {'A': ['EEG A1', 'EEG A2', 'EEG A3', 'EEG A4']}}}}
        document['sharing'] = {'preparation_cache_mib': cache}
        destination = tmp_path/'experiment.yaml'
        destination.write_text(yaml.safe_dump(document))
        return destination
    return make


def test_group_defaults_and_preflight(experiment_file):
    path = experiment_file()
    experiment = load_experiment(path)
    groups = dict(experiment.groups)
    assert groups['normalized'].zscore and not groups['amplitude'].zscore
    assert groups['amplitude'].detrend
    assert groups['normalized'].reference['method'] == 'car'
    assert groups['shaft'].reference['method'] == 'shaft_average'
    assert len({cfg.run_name for _, cfg in experiment.groups}) == 3
    report = preflight_experiment(path)
    assert report['output_count_per_recording'] == 6
    assert set(report['groups']) == {'normalized', 'amplitude', 'shaft'}


@pytest.mark.parametrize('workers', [1, 2])
def test_shared_vs_separate_outputs_and_reuse(experiment_file, workers):
    path = experiment_file(workers=workers)
    experiment = load_experiment(path)
    with patch('pyspi.workflow._shared_prepare.prepare_recording', wraps=prepare_recording) as loading:
        result = run_experiment(experiment, max_windows=2)
        assert loading.call_count == 1
    assert result['preparation_cache'][0]['base_misses'] == 1
    assert result['preparation_cache'][0]['reference_hits'] == 1
    for record, (_, cfg) in zip(result['runs'], experiment.groups):
        shared = open_run(record['path'])
        if workers > 1:
            import json
            session = json.loads((Path(record['path'])/'run.json').read_text())['sessions'][-1]
            assert session['worker_ready_delay_seconds']
            assert min(session['worker_ready_delay_seconds'].values()) > 0
            assert session['result_delivery_seconds'] >= 0
            assert session['worker_calculation_seconds'] > 0
        [separate_path] = run(replace(cfg, run_name=cfg.run_name+'-separate'), max_windows=2)
        separate = open_run(separate_path)
        assert shared.recording['reference'] == separate.recording['reference']
        assert shared.recording['preprocessing'] == separate.recording['preprocessing']
        for metric in shared.metrics:
            assert set(shared.statuses(metric)) == {'success'}
            np.testing.assert_allclose(shared.load(metric), separate.load(metric), atol=2e-7, rtol=1e-6)
    with patch('pyspi.workflow._shared_prepare.prepare_recording', side_effect=AssertionError('complete resume must not prepare')):
        resumed = run_experiment(experiment, max_windows=2, resume=True)
    assert resumed['runs'] == result['runs']
    with pytest.raises(WorkflowError, match='already exists'):
        run_experiment(experiment, max_windows=2)


def test_partial_group_resume_starts_missing_groups(experiment_file):
    experiment = load_experiment(experiment_file())
    first = replace(experiment, groups=experiment.groups[:1])
    done = run_experiment(first, max_windows=1)
    resumed = run_experiment(experiment, max_windows=1, resume=True)
    assert resumed['runs'][0] == done['runs'][0]
    assert len(resumed['runs']) == 3


def test_cache_budget_and_input_setting_isolation(experiment_file):
    experiment = load_experiment(experiment_file())
    cfg = preflight(experiment.groups[0][1]).configuration
    recording = cfg.recordings[0]
    cache = SharedPreparation(max_bytes=1, band_unions={})
    first = cache(recording, cfg, bands=((30., 70.),))
    assert cache.bytes == 0 and not cache.entries
    assert isinstance(first.data, np.memmap)
    assert not first.data.flags.writeable
    key = preparation_key(recording, cfg)
    assert preparation_key(recording, replace(cfg, zscore=False)) == key
    assert preparation_key(recording, replace(cfg, bandpass=(3, 200))) != key
    assert preparation_key(replace(recording, channels=tuple(reversed(recording.channels or ()))), cfg) != key
    cache.close()


def test_invalid_group_override_prevents_any_output(experiment_file):
    path = experiment_file()
    document = yaml.safe_load(path.read_text())
    document['groups']['bad'] = {'preprocessing': {'zsocre': True}}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(WorkflowError, match='unknown fields'):
        run_experiment(path, max_windows=1)


def test_adaptive_shared_bands_match_reference_first(experiment_file):
    from pyspi.workflow._adaptive_reference import build_adaptive_reference
    from pyspi.workflow._prepare import reference_names
    experiment = load_experiment(experiment_file())
    cfg = preflight(experiment.groups[0][1]).configuration
    names = ['EEG A1', 'EEG A2', 'EEG A3', 'EEG A4']
    reference = build_adaptive_reference(names, ['EEG A']*4, patient_id=cfg.recordings[0].patient_id)
    cfg = replace(cfg, reference=reference)
    recording = cfg.recordings[0]
    cache = SharedPreparation(max_bytes=8*2**20, band_unions={})
    shared = cache(recording, cfg, bands=((30., 70.),))
    independent = prepare_recording(recording, cfg, bands=((30., 70.),))
    assert shared.channel_names == reference_names(names, reference)
    np.testing.assert_allclose(shared.data, independent.data, atol=1e-20, rtol=2e-12)
    np.testing.assert_allclose(shared.bands[(30., 70.)], independent.bands[(30., 70.)], atol=1e-20, rtol=2e-12)
    assert shared.details['reference_geometry'] == independent.details['reference_geometry']
    cache.close()


def test_group_profile_includes_all_groups_and_process_memory(experiment_file, tmp_path):
    from pyspi.workflow._profile import profile_experiment
    report = profile_experiment(experiment_file(), max_windows=1, report_path=tmp_path/'profile.json')
    assert report['recording_windows'] == 1 and report['group_windows'] == 3
    assert len(report['runs']) == 3
    assert report['process_tree_memory']['peak_rss_mib'] > 0
    assert report['preparation_cache'][0]['base_misses'] == 1
