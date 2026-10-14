"""Integrity detection and salvage, without prescribing a future partial-load API."""

import json
import runpy
import shutil

import numpy as np
import pytest

from storage_cases import assert_bits, data_files, matrix_case, tree_fingerprint


@pytest.fixture
def sharded_output(store, tmp_path, codec):
    data = matrix_case("sym", windows=13)
    store.write_metric(tmp_path, "damaged", data, codec=codec,
                       target_chunk_values=20, max_shard_values=40)
    store.write_metric(tmp_path, "healthy", data + 10, codec=codec)
    return tmp_path, data


@pytest.mark.parametrize("damage", ["flip_payload", "truncate", "flip_index"])
def test_corruption_names_metric_and_keeps_other_shards_readable(store, sharded_output, damage):
    path, data = sharded_output
    files = data_files(path / "damaged" / "upper")
    assert len(files) > 1
    first = next(p for p in files if p.name == "0")
    raw = bytearray(first.read_bytes())
    if damage == "truncate":
        raw = raw[:len(raw) // 2]
    else:
        raw[0 if damage == "flip_payload" else -1] ^= 0x10
    first.write_bytes(raw)
    errors = store.verify(path)
    assert set(errors) == {"damaged"}
    assert errors["damaged"]
    reader = store.open_results(path)
    assert_bits(reader.load("healthy"), data + 10)
    # Chunks span two windows and shards span four: the last window is independent.
    assert_bits(reader.load("damaged", windows=12), data[:, :, 12])


def test_invalid_json_metadata_is_identified(store, sharded_output):
    path, data = sharded_output
    (path / "damaged" / "zarr.json").write_text("{invalid JSON")
    errors = store.verify(path)
    assert set(errors) == {"damaged"}
    assert_bits(store.open_results(path).load("healthy"), data + 10)


def test_missing_data_shard_is_identified(store, sharded_output):
    path, _ = sharded_output
    data_files(path / "damaged" / "upper")[0].unlink()
    assert "damaged" in store.verify(path)


def test_missing_residual_is_identified(store, force_mode, tmp_path, codec):
    force_mode("sym_res")
    store.write_metric(tmp_path, "residual", matrix_case("sym_res"), codec=codec)
    for p in data_files(tmp_path / "residual" / "residual"):
        p.unlink()
    assert "residual" in store.verify(tmp_path)


def test_deleted_metric_directory_is_identified(store, sharded_output):
    path, _ = sharded_output
    shutil.rmtree(path / "damaged")
    assert "damaged" in store.verify(path)


def test_interruption_before_data_write_is_identified(store, tmp_path, monkeypatch):
    import zarr

    setitem = zarr.Array.__setitem__

    def interrupt(array, selection, values):
        raise OSError("simulated interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(zarr.Array, "__setitem__", interrupt)
        with pytest.raises(OSError, match="simulated interrupted write"):
            store.write_metric(tmp_path, "interrupted", matrix_case("sym"))
    assert zarr.Array.__setitem__ is setitem
    assert "interrupted" in store.verify(tmp_path)


def test_valid_json_with_inconsistent_channel_count_is_identified(store, sharded_output):
    path, _ = sharded_output
    metadata_path = path / "damaged" / "zarr.json"
    meta = json.loads(metadata_path.read_text())
    meta["attributes"]["pspi"]["n"] += 1
    metadata_path.write_text(json.dumps(meta))
    assert "damaged" in store.verify(path)


def test_copy_corruption_does_not_modify_source(store, sharded_output, tmp_path_factory):
    source, data = sharded_output
    before = tree_fingerprint(source)
    copied = tmp_path_factory.mktemp("copy") / "recording.pspi"
    shutil.copytree(source, copied)
    target = data_files(copied / "damaged" / "upper")[0]
    raw = bytearray(target.read_bytes())
    raw[0] ^= 1
    target.write_bytes(raw)
    assert "damaged" in store.verify(copied)
    assert store.verify(source) == {}
    assert tree_fingerprint(source) == before
    assert_bits(store.open_results(source).load("damaged"), data)


@pytest.mark.parametrize("case", ["usage", "clean", "corrupt"])
def test_verification_command_reports_failures_and_exit_status(store, tmp_path, monkeypatch, capsys, case):
    arguments = [store.__file__]
    if case != "usage":
        store.write_metric(tmp_path, "metric", matrix_case("sym"))
        arguments.extend(["verify", str(tmp_path)])
    if case == "corrupt":
        target = data_files(tmp_path / "metric" / "upper")[0]
        raw = bytearray(target.read_bytes())
        raw[0] ^= 1
        target.write_bytes(raw)
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(store.__file__, run_name="__main__")
    if case == "usage":
        assert "usage:" in str(exited.value.code)
    else:
        output = capsys.readouterr().out
        assert exited.value.code == (0 if case == "clean" else 1)
        assert ("all metrics verified" if case == "clean" else "FAILED metric:") in output
