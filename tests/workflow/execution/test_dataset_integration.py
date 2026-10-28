"""Metadata-driven shared dataset output across references and precisions."""
from dataclasses import replace
import json

import numpy as np
import pytest

from seeg_dataset import create_patient, list_runs, open_run, open_patient
from pyspi.workflow import load_config, preflight, run, WorkflowError
from pyspi.workflow._dataset import dataset_address


def dataset(tmp_path, *, contacts=("B2", "A1"), labels=("EEG A1", "EEG B2")):
    root = tmp_path / "lake"
    patient = root / "example.zarr"
    create_patient(patient, "example", arrays={"electrode": list(contacts), "target": [False, True]},
                   groups={"edf_history": {"contact": ["A1", "B2"], "edf_label": list(labels),
                                           "used": [True, True]}})
    return root, patient


def config(execution_config, source, root, **changes):
    return execution_config(source=source, recordings=[{"path": str(source), "patient_id": "example",
                                                       "recording_id": "baseline"}],
                            dataset={"directory": str(root)}, output_directory=None,
                            output_dtype="float32", **changes)


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_authoritative_order_shared_reader_and_distinct_references(execution_config, signal_edf, tmp_path):
    root, patient = dataset(tmp_path)
    before = snapshot(patient / "patient_info")
    cfg = load_config(config(execution_config, signal_edf(), root, reference="none"))
    report = preflight(cfg)
    assert report.recordings[0]["output_channels"] == ["B2", "A1"]
    assert [c["source_index"] for c in report.recordings[0]["channels"]] == [1, 0]
    assert not (patient / "connectivity").exists()
    [none] = run(cfg, max_windows=2)
    [car] = run(replace(cfg, reference={"method": "car"}), max_windows=2)
    [amplitude] = run(replace(cfg, zscore=False), max_windows=2)
    assert len(set([none, car, amplitude])) == 3
    assert set(list_runs(patient)) == {none, car, amplitude}
    assert none.relative_to(patient).parts[:4] == ("connectivity", "baseline", "0p5s-step0p5s", "none")
    reader = open_run(none)
    assert reader.recording["channels"] == ["B2", "A1"]
    assert reader.recording["reference"] == {"method": "none"}
    assert reader.load(reader.metrics[0]).dtype == np.float32
    assert reader.recording["channel_mapping"] == {"B2": ["B2"], "A1": ["A1"]}
    assert before == snapshot(patient / "patient_info")
    with pytest.raises(WorkflowError, match="output already exists"):
        run(cfg, max_windows=2)
    with pytest.raises(WorkflowError, match="incompatible resume"):
        run(replace(cfg, output_dtype="float64"), resume=True, max_windows=2)
    run(cfg, resume=True, max_windows=2)


def test_lake_config_resolves_destination_without_building(execution_config, signal_edf, tmp_path):
    root, patient = dataset(tmp_path)
    lake = tmp_path / "data-lake.toml"
    lake.write_text('dataset_dir = "lake"\n')
    path = config(execution_config, signal_edf(), root)
    import yaml
    document = yaml.safe_load(path.read_text())
    document["dataset"] = {"data_lake_config": str(lake)}
    path.write_text(yaml.safe_dump(document))
    report = preflight(path)
    assert report.configuration.output_directory == root
    assert not (patient / "connectivity").exists()
    lake.write_text('# corrected destination not chosen\n')
    with pytest.raises(WorkflowError, match="dataset_dir"):
        load_config(path)


def test_inline_order_conflicts_and_bad_history_stop_before_output(execution_config, signal_edf, tmp_path):
    root, patient = dataset(tmp_path)
    source = signal_edf()
    cfg = load_config(config(execution_config, source, root))
    recording = replace(cfg.recordings[0], channels=({"index": 0, "name": "A1"}, {"index": 1, "name": "B2"}))
    with pytest.raises(WorkflowError, match="authoritative.*order"):
        preflight(replace(cfg, recordings=(recording,)))
    import zarr
    zarr.open_group(patient, mode="r+")["patient_info/edf_history/edf_label"][:] = ["missing", "EEG B2"]
    with pytest.raises(WorkflowError, match="0 EDF matches"):
        run(cfg, max_windows=1)
    assert not (patient / "connectivity").exists()


def test_mixed_dtypes_and_masked_reference_contacts(execution_config, signal_edf, metric_config, tmp_path):
    root = tmp_path / "lake"
    patient = root / "example.zarr"
    names = ["A1", "A2", "A3", "A4"]
    create_patient(patient, "example", arrays={"electrode": names})
    source = signal_edf(labels=tuple("EEG " + n for n in names))
    metrics = metric_config({"Covariance": {"configs": [{"estimator": "EmpiricalCovariance"}]},
                             "SpearmanR": {"configs": [{"squared": False}]}})
    cfg = load_config(config(execution_config, source, root,
                             metrics={"config": str(metrics)},
                             reference={"method": "laplacian", "shafts": {"A": names}}))
    specs = preflight(cfg).metrics
    cfg = replace(cfg, output_dtype="float16", output_dtype_overrides={specs[1].identifier: "float64"})
    [path] = run(cfg, max_windows=2)
    reader = open_run(path)
    assert reader.load(specs[0].identifier).dtype == np.float16
    assert reader.load(specs[1].identifier).dtype == np.float64
    assert reader.recording["reference_available"] == [False, True, True, False]
    assert np.isnan(reader.load(specs[0].identifier)[[0, 3]]).all()
    assert list(reader.statuses(specs[0].identifier)) == ["success", "success"]


def test_bipolar_outputs_have_explicit_contact_mapping(execution_config, signal_edf, tmp_path):
    root=tmp_path/'lake'; patient=root/'example.zarr'
    names=['A1','A2','B1','B2']
    create_patient(patient,'example',arrays={'electrode':names})
    source=signal_edf(labels=tuple('EEG '+n for n in names))
    cfg=load_config(config(execution_config,source,root,
        reference={'method':'bipolar','anode':['A1','B1'],'cathode':['A2','B2'],
                   'names':['A1-A2','B1-B2']}))
    [path]=run(cfg,max_windows=1)
    reader=open_run(path)
    assert reader.recording['channels']==['A1-A2','B1-B2']
    assert reader.recording['channel_mapping']=={'A1-A2':['A1','A2'],'B1-B2':['B1','B2']}
    assert reader.load(reader.metrics[0]).shape==(2,2,1)
    assert open_patient(patient)['patient_info/electrode'][:].tolist()==names


def test_calibration_uses_resolved_metadata_selection(execution_config, signal_edf, tmp_path):
    from pyspi.workflow._calibrate import calibrate
    root, patient = dataset(tmp_path)
    source = signal_edf(labels=('EEG A1','EEG B2','EEG excluded'))
    path = config(execution_config,source,root,reference='none')
    result = calibrate(path,channels=[2],windows_per_point=1)
    assert result['source']['output_channels'] == 2
    assert result['preparation']['details']['native_rate_groups'][0]['channels'] == ['B2','A1']
    assert not (patient/'connectivity').exists()


@pytest.mark.parametrize('problem', ['ambiguous_header', 'duplicate_source', 'history_exclusion'])
def test_inconsistent_metadata_mappings_fail_before_output(problem, execution_config, signal_edf, tmp_path):
    import zarr
    root, patient = dataset(tmp_path)
    labels = ('EEG A1', 'EEG B2', 'EEG B2') if problem == 'ambiguous_header' else ('EEG A1', 'EEG B2')
    source = signal_edf(labels=labels)
    info = zarr.open_group(patient, mode='r+')['patient_info']
    if problem == 'duplicate_source':
        info['edf_history/edf_label'][:] = ['EEG A1', 'EEG A1']
    elif problem == 'history_exclusion':
        info['edf_history/used'][:] = [True, False]
    path = config(execution_config, source, root)
    with pytest.raises(WorkflowError, match='EDF matches|same EDF signal|history contacts differ'):
        run(path, max_windows=1)
    assert not (patient/'connectivity').exists()
