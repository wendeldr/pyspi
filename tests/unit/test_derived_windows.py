"""Saved-graph transformations, missing-data alignment and bounded reads."""
import numpy as np
import pytest

from pyspi.workflow.derived import iter_derived_windows
from pyspi.workflow import WorkflowError


class Reader:
    path = '/tmp/synthetic-reader/car/normalized/run'
    def __init__(self):
        self.values = np.array([[[0, 0, 0, 0, 0, 0], [.1, 1.8, .6, .2, np.nan, .9]],
                                [[.1, 1.8, .6, .2, np.nan, .9], [0, 0, 0, 0, 0, 0]]])
        self.spec = {'module': 'pyspi.statistics.distance', 'class_name': 'PairwiseDistance',
                     'parameters': {'metric': 'cosine'}, 'identifier': 'pdist_cosine'}
        self.recording = {'preprocessing': {'detrend': True, 'zscore': True},
                          'reference': {'method': 'car'}, 'windows': {'step_samples': 1024}}
        self.status = np.full(6, 'success', dtype='U10')
        self.reads = []
    def info(self, metric):
        assert metric == 'pdist_cosine'
        return {'metric': self.spec, 'n': 2, 'windows': 6}
    def statuses(self, metric):
        return self.status
    def load(self, metric, windows, return_availability):
        self.reads.append(windows.copy())
        value = self.values[:, :, windows]
        return value, np.ones_like(value, dtype=bool)


def test_pearson_and_absolute_happen_before_temporal_summary():
    reader = Reader()
    signed = list(iter_derived_windows(reader, 'pdist_cosine', transform='pearson', windows=[0, 1]))[0]
    absolute = list(iter_derived_windows(reader, 'pdist_cosine', transform='absolute_pearson', windows=[0, 1]))[0]
    np.testing.assert_allclose(signed.values[0, 1], [.9, -.8])
    assert np.mean(absolute.values[0, 1]) == pytest.approx(.85)
    assert abs(np.mean(signed.values[0, 1])) == pytest.approx(.05)
    assert absolute.provenance['recording']['reference'] == {'method': 'car'}
    assert absolute.provenance['temporal_summary'] is None
    assert np.isnan(absolute.values[0, 0]).all()


def test_changes_cross_batch_boundaries_but_not_missing_windows():
    reader = Reader()
    batches = list(iter_derived_windows(reader, 'pdist_cosine', transform='absolute_pearson',
                                       absolute_change=True, windows=[0, 1, 3, 4, 5], batch_windows=1))
    actual = np.concatenate([batch.values for batch in batches], axis=2)
    np.testing.assert_allclose(actual[0, 1], [np.nan, .1, np.nan, np.nan, np.nan], atol=1e-15)
    assert max(len(read) for read in reader.reads) == 1
    np.testing.assert_equal(np.concatenate([batch.window_indices for batch in batches]), [0, 1, 3, 4, 5])


def test_failed_windows_do_not_bridge_even_if_payload_is_finite():
    reader = Reader()
    reader.status[1] = 'failed'
    result = list(iter_derived_windows(reader, 'pdist_cosine', absolute_change=True, windows=[0, 1, 2, 3]))[0]
    np.testing.assert_allclose(result.values[0, 1], [np.nan, np.nan, np.nan, .4])
    assert not result.valid[0, 1, :3].any()


@pytest.mark.parametrize('indices', [[1, 0], [0, 0], [-1, 1], [0., 1.], [0, 6]])
def test_bad_window_selections_rejected(indices):
    with pytest.raises(WorkflowError, match='indices'):
        list(iter_derived_windows(Reader(), 'pdist_cosine', windows=indices))


def test_cannot_derive_from_unverified_centering_or_different_metric():
    reader = Reader()
    reader.recording['preprocessing']['zscore'] = False
    with pytest.raises(WorkflowError, match='z-scoring'):
        list(iter_derived_windows(reader, 'pdist_cosine', transform='pearson'))
    reader.recording['preprocessing']['zscore'] = True
    reader.spec['parameters']['metric'] = 'euclidean'
    with pytest.raises(WorkflowError, match='cosine'):
        list(iter_derived_windows(reader, 'pdist_cosine', transform='pearson'))


def test_all_saved_metrics_can_use_identity_changes_without_normalization():
    reader = Reader()
    reader.recording['preprocessing']['zscore'] = False
    reader.spec['parameters']['metric'] = 'euclidean'
    result = list(iter_derived_windows(reader, 'pdist_cosine', absolute_change=True, windows=[0, 1]))[0]
    assert result.values[0, 1, 1] == pytest.approx(1.7)
