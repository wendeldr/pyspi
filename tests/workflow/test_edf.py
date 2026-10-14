from fractions import Fraction
from pathlib import Path

import pytest

from pyspi.workflow import WorkflowError
from pyspi.workflow._edf import read_header, select_channels


def test_header_reads_only_fixed_and_signal_headers(edf, monkeypatch):
    path = edf(labels=("EEG A", "EEG B", "EDF Annotations"), samples=(8, 4, 16),
               records=2, duration="0.5", reserved="EDF+C")
    original = Path.open
    reads = []

    class Guard:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, size):
            reads.append((self.stream.tell(), size))
            assert self.stream.tell() + size <= 256 + 3 * 256
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs:
                        Guard(original(self, *args, **kwargs)) if self == path else original(self, *args, **kwargs))
    header = read_header(path)
    assert reads == [(0, 256), (256, 768)]
    assert header.record_seconds == Fraction(1, 2) and header.records == 2
    assert [signal.rate for signal in header.signals] == [16, 8, 32]
    assert header.signals[2].annotation


def test_unknown_record_count_inferred(edf):
    header = read_header(edf(records=5, declared_records=-1))
    assert header.records == 5


@pytest.mark.parametrize("offset,size,value,match", [
    (0, 8, b"BIOSEMI", "not a complete EDF"),
    (184, 8, b"255", "signal header"),
    (192, 44, b"EDF+D", "discontinuous"),
    (236, 8, b"-2", "invalid EDF"),
    (236, 8, b"4", "geometry"),
    (244, 8, b"0", "invalid EDF"),
    (244, 8, b"NaN", "invalid EDF"),
    (244, 8, b"nonsense", "invalid EDF header"),
    (252, 4, b"0", "invalid EDF"),
    (256 + 216 * 2, 8, b"0", "samples per record"),
    (256, 16, b"\xff", "invalid EDF header"),
])
def test_invalid_headers_rejected(edf, offset, size, value, match):
    path = edf()
    content = bytearray(path.read_bytes())
    content[offset:offset + size] = value.ljust(size)
    path.write_bytes(content)
    with pytest.raises(WorkflowError, match=match):
        read_header(path)


@pytest.mark.parametrize("truncate", [1, 100, 600])
def test_truncated_file_rejected(edf, truncate):
    path = edf()
    path.write_bytes(path.read_bytes()[:-truncate])
    with pytest.raises(WorkflowError):
        read_header(path)


def test_missing_input_and_short_header(tmp_path):
    path = tmp_path / "absent.EDF"
    with pytest.raises(WorkflowError, match="unreadable"):
        read_header(path)
    path.write_bytes(b"0")
    with pytest.raises(WorkflowError, match="complete EDF"):
        read_header(path)


def test_default_eeg_and_explicit_order_mapping(edf):
    path = edf(labels=("ECG ECG1", "SEEG A1", "B2", "EDF Annotations"), samples=(8, 8, 8, 16))
    header = read_header(path)
    assert [(s.index, name) for s, name in select_channels(header)] == [(1, "SEEG A1"), (2, "B2")]
    selected = select_channels(header, [{"index": 2, "name": "second"},
                                        {"label": "SEEG A1", "name": "first"}])
    assert [(s.index, s.label, name) for s, name in selected] == [(2, "B2", "second"), (1, "SEEG A1", "first")]


@pytest.mark.parametrize("selection,match", [
    (["absent"], "0 matches"), ([True], "selectors must"), ([1.0], "selectors must"),
    ([-1], "outside"), ([10], "outside"), ([None], "selectors must"),
    ([{"index": 0, "label": "EEG B2"}], "exactly one"),
    ([{"index": 0, "other": 1}], "unknown fields"),
    ([{"index": 0, "name": ""}], "nonempty string"),
    ([0, "EEG B2"], "duplicate"),
    ([{"index": 0, "name": "same"}, {"index": 1, "name": "same"}], "duplicate"),
])
def test_invalid_selections_fail(edf, selection, match):
    with pytest.raises(WorkflowError, match=match):
        select_channels(read_header(edf()), selection)


def test_duplicate_and_empty_labels_require_disambiguation(edf):
    header = read_header(edf(labels=("A", "A", ""), samples=(8, 8, 8)))
    with pytest.raises(WorkflowError, match="2 matches"):
        select_channels(header, ["A"])
    with pytest.raises(WorkflowError, match="duplicate"):
        select_channels(header)
    with pytest.raises(WorkflowError, match="empty EDF"):
        select_channels(header, [2])
    assert [name for _, name in select_channels(header, [{"index": 1, "name": "A"},
                                                       {"index": 2, "name": "unnamed"}])] == ["A", "unnamed"]


def test_annotations_cannot_be_selected(edf):
    header = read_header(edf(labels=("EDF Annotations",), samples=(8,)))
    with pytest.raises(WorkflowError, match="no EEG"):
        select_channels(header)
    with pytest.raises(WorkflowError, match="annotation"):
        select_channels(header, [0])
