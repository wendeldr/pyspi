"""Metadata-resolved shaft means preserve patient axes and explicit layouts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyspi.workflow._adaptive_reference import resolve_reference
from pyspi.workflow._config import WorkflowError, reference_config
from pyspi.workflow._reference import apply_shaft_reference, reference_plan


def test_metadata_shaft_average_and_patient_isolation(monkeypatch, tmp_path):
    import seeg_dataset
    names = ['B2', 'A3', 'B1', 'A1', 'C1']
    groups = ['B', 'A', 'B', 'A', 'C']
    monkeypatch.setattr(seeg_dataset, 'patient_contacts', lambda path, patient_id: names)
    monkeypatch.setattr(seeg_dataset, 'open_patient', lambda path: {'patient_info': {'electrodegroup': np.array(groups)}})
    reference = reference_config({'method': 'shaft_average', 'shafts': 'metadata'}, tmp_path)
    cfg = SimpleNamespace(reference=reference, dataset_directory=tmp_path)
    recording = SimpleNamespace(patient_id='001')
    resolved = resolve_reference(cfg, recording)
    assert resolved['shafts'] == {'B': ['B2', 'B1'], 'A': ['A3', 'A1'], 'C': ['C1']}
    assert resolved['patient_id'] == '001'
    x = np.arange(25.).reshape(5, 5)
    plan = reference_plan(names, resolved)
    actual = apply_shaft_reference(x, names, resolved, plan)
    np.testing.assert_equal(actual[0], x[0]-(x[0]+x[2])/2)
    np.testing.assert_equal(actual[1], x[1]-(x[1]+x[3])/2)
    assert not plan[1][4]  # single-contact shaft has no defined shaft-centered signal
    assert plan[2][4] == 'insufficient_shaft_contacts'
    cfg.reference = resolved
    assert resolve_reference(cfg, recording) is resolved
    with pytest.raises(WorkflowError, match='different patient'):
        resolve_reference(cfg, SimpleNamespace(patient_id='002'))


def test_explicit_layout_keeps_existing_mean_definition():
    reference = reference_config({'method': 'shaft_average', 'shafts': {'A': ['A1', 'A2']}}, Path('.'))
    assert resolve_reference(SimpleNamespace(reference=reference, dataset_directory=None),
                             SimpleNamespace(patient_id='001')) is reference


def test_automatic_layout_requires_authoritative_metadata():
    reference = {'method': 'shaft_average', 'shafts': 'metadata'}
    with pytest.raises(WorkflowError, match='requires dataset'):
        resolve_reference(SimpleNamespace(reference=reference, dataset_directory=None),
                          SimpleNamespace(patient_id='001'))
    with pytest.raises(WorkflowError, match='physically ordered'):
        reference_config({'method': 'laplacian', 'shafts': 'metadata'}, Path('.'))
