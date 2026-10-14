"""Fault injection against the completion contract and entry-level recovery API."""

import json
import shutil
import struct
import warnings

import google_crc32c
import numpy as np
import pytest
import zarr

from storage_cases import assert_bits, data_files, matrix_case, special_values, tree_fingerprint


def manifest(path):
    return json.loads((path / "pspi-manifest.json").read_text())


def edit_index(array_path, change, shard=0):
    """Independently edit an on-disk shard index and recompute its checksum."""
    meta = json.loads((array_path / "zarr.json").read_text())
    config = meta["codecs"][0]["configuration"]
    slots = meta["chunk_grid"]["configuration"]["chunk_shape"][0] // config["chunk_shape"][0]
    index_size = slots * 16 + 4
    path = array_path / "c" / str(shard) / "0"
    raw = bytearray(path.read_bytes())
    index = np.frombuffer(raw[-index_size:-4], dtype="<u8").copy().reshape(slots, 2)
    change(index, len(raw) - index_size)
    encoded = index.tobytes()
    raw[-index_size:] = encoded + struct.pack("<I", google_crc32c.value(encoded))
    path.write_bytes(raw)


def remove_inner(array_path, slot=0, shard=0):
    def missing(index, end):
        index[slot] = np.iinfo(np.uint64).max
    edit_index(array_path, missing, shard)


def expect_partial(store, reader, metric, original, expected_mask, windows=None):
    selected = original if windows is None else original[:, :, windows]
    with pytest.warns(store.PartialReadWarning, match=metric) as caught:
        values, available = reader.load(metric, windows, return_availability=True)
    assert len(caught) == 1
    assert values.shape == selected.shape == available.shape
    assert available.dtype == np.bool_
    np.testing.assert_array_equal(available, expected_mask)
    assert np.isnan(values[~available]).all()
    assert_bits(values[available], selected[available])
    with pytest.warns(store.PartialReadWarning):
        assert_bits(reader.load(metric, windows), values)
    return str(caught[0].message)


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_missing_primary_preserves_positions_and_exact_survivors(store, force_mode, tmp_path, codec, kind, dtype):
    force_mode(kind)
    data = matrix_case(kind, windows=9, dtype=dtype)
    primary_columns = 16 if kind == "full" else 10
    store.write_metric(tmp_path, "metric", data, dtype=dtype, codec=codec,
                       target_chunk_values=primary_columns * 2, max_shard_values=primary_columns * 4)
    assert store.open_results(tmp_path).info("metric")["mode"] == kind
    key = "full" if kind == "full" else "upper"
    remove_inner(tmp_path / "metric" / key)
    mask = np.ones(data.shape, dtype=bool)
    mask[:, :, :2] = False
    reader = store.open_results(tmp_path)
    message = expect_partial(store, reader, "metric", data, mask)
    assert key in message and "[0, 2)" in message and "inner chunk is missing" in message
    for selection in ([8, 0, 8, 1, 4], -9, [0]):
        expect_partial(store, reader, "metric", data, mask[:, :, selection], selection)
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.PartialReadWarning)
        values, available = reader.load("metric", [8, 4, 8], return_availability=True)
        assert_bits(values, data[:, :, [8, 4, 8]])
        assert available.all()
        values, available = reader.load("metric", [], return_availability=True)
        assert values.shape == available.shape == (4, 4, 0)
    assert "metric" in store.verify(tmp_path)


@pytest.mark.parametrize("kind", ["sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_missing_residual_retains_upper_and_diagonal(store, force_mode, tmp_path, codec, kind, dtype):
    force_mode(kind)
    data = matrix_case(kind, windows=9, dtype=dtype)
    store.write_metric(tmp_path, "metric", data, dtype=dtype, codec=codec,
                       target_chunk_values=20, max_shard_values=40)
    remove_inner(tmp_path / "metric" / "residual")  # three windows per residual chunk
    mask = np.ones(data.shape, dtype=bool)
    i, j = np.tril_indices(4, -1)
    mask[i[:, None], j[:, None], np.arange(3)[None, :]] = False
    message = expect_partial(store, store.open_results(tmp_path), "metric", data, mask)
    assert "residual" in message and "[0, 3)" in message
    for selection in ([8, 0, 1, 8], 0):
        expect_partial(store, store.open_results(tmp_path), "metric", data, mask[:, :, selection], selection)


@pytest.mark.parametrize("damage", ["payload", "index_crc", "offset", "missing_shard", "truncated_index"])
def test_corruption_recovery_is_bounded_by_checksums(store, tmp_path, codec, damage):
    data = matrix_case("sym", windows=9)
    store.write_metric(tmp_path, "metric", data, codec=codec,
                       target_chunk_values=20, max_shard_values=40)
    array_path = tmp_path / "metric" / "upper"
    file = array_path / "c" / "0" / "0"
    raw = bytearray(file.read_bytes())
    if damage == "payload":
        raw[0] ^= 0x10
        file.write_bytes(raw)
    elif damage == "index_crc":
        raw[-1] ^= 0x10
        file.write_bytes(raw)
    elif damage == "offset":
        edit_index(array_path, lambda index, end: index.__setitem__((0, 0), end + 10))
    elif damage == "missing_shard":
        file.unlink()
    else:
        file.write_bytes(b"truncated")
    mask = np.ones(data.shape, dtype=bool)
    mask[:, :, :(2 if damage in ("payload", "offset") else 4)] = False
    expect_partial(store, store.open_results(tmp_path), "metric", data, mask)
    assert "metric" in store.verify(tmp_path)


def test_out_of_shape_padded_slots_are_not_missing_data(store, tmp_path, codec):
    data = matrix_case("sym", windows=5)
    store.write_metric(tmp_path, "metric", data, codec=codec,
                       target_chunk_values=20, max_shard_values=40)
    remove_inner(tmp_path / "metric" / "upper", slot=1, shard=1)  # window six does not exist
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.PartialReadWarning)
        values, available = store.open_results(tmp_path).load("metric", return_availability=True)
    assert_bits(values, data)
    assert available.all()
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_saved_nan_inf_and_zero_chunks_are_available_without_warning(store, tmp_path, codec, dtype):
    data = np.resize(special_values(dtype), (4, 4, 9)).copy()
    data[:, :, :2] = 0
    store.write_metric(tmp_path, "special", data, codec=codec, dtype=dtype,
                       target_chunk_values=32, max_shard_values=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.PartialReadWarning)
        values, available = store.open_results(tmp_path).load("special", return_availability=True)
    assert_bits(values, data)
    assert available.all()
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("damage", ["delete_metric", "delete_array_metadata", "change_array_metadata", "metric_json", "root_json"])
def test_metadata_recovery_and_read_only_behavior(store, tmp_path, codec, damage):
    data = matrix_case("sym")
    store.write_metric(tmp_path, "metric", data, codec=codec)
    if damage == "delete_metric":
        shutil.rmtree(tmp_path / "metric")
    elif damage == "delete_array_metadata":
        (tmp_path / "metric" / "upper" / "zarr.json").unlink()
    elif damage == "change_array_metadata":
        path = tmp_path / "metric" / "upper" / "zarr.json"
        value = json.loads(path.read_text())
        value["fill_value"] = 3
        path.write_text(json.dumps(value))
    else:
        path = tmp_path / ("metric/zarr.json" if damage == "metric_json" else "zarr.json")
        path.write_text("[]")
    before = tree_fingerprint(tmp_path)
    mask = np.ones(data.shape, dtype=bool) if damage.endswith("json") else np.zeros(data.shape, dtype=bool)
    reader = store.open_results(tmp_path)
    assert reader.metrics == ["metric"]
    expect_partial(store, reader, "metric", data, mask)
    assert store.verify(tmp_path)
    assert tree_fingerprint(tmp_path) == before


def test_single_matrix_partial_shape(store, tmp_path, codec):
    data = matrix_case("sym", windows=1)[:, :, 0]
    store.write_metric(tmp_path, "metric", data, codec=codec)
    data_files(tmp_path / "metric" / "upper")[0].unlink()
    with pytest.warns(store.PartialReadWarning):
        values, available = store.open_results(tmp_path).load("metric", return_availability=True)
    assert values.shape == available.shape == (4, 4)
    assert np.isnan(values).all() and not available.any()


def test_batch_inventory_survives_interruption(store, tmp_path, monkeypatch):
    data = matrix_case("sym")
    original = zarr.Array.__setitem__

    def interrupt(array, selection, values):
        current = manifest(tmp_path)
        assert set(current["metrics"]) == {"alpha", "beta", "gamma"}
        entry = current["metrics"][array.path.split("/")[0]]
        assert entry["state"] == "writing"
        assert set(entry["array_metadata"]) == set(entry["meta"]["arrays"])
        if array.path.startswith("beta/"):
            raise OSError("interrupted batch")
        return original(array, selection, values)

    monkeypatch.setattr(zarr.Array, "__setitem__", interrupt)
    with pytest.raises(OSError, match="interrupted batch"):
        store.write_results(tmp_path, dict(alpha=data, beta=data, gamma=data))
    assert {k: v["state"] for k, v in manifest(tmp_path)["metrics"].items()} == {
        "alpha": "complete", "beta": "writing", "gamma": "pending"}
    assert set(store.verify(tmp_path)) == {"beta", "gamma"}
    reader = store.open_results(tmp_path)
    assert_bits(reader.load("alpha"), data)
    for metric in ("beta", "gamma"):
        expect_partial(store, reader, metric, data, np.zeros(data.shape, dtype=bool))


@pytest.mark.parametrize("stage", ["pack", "payload", "flush", "publish", "publish_directory_flush"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_write_never_claims_complete(store, tmp_path, monkeypatch, stage, overwrite):
    import store_integrity

    data = matrix_case("sym")
    if overwrite:
        store.write_metric(tmp_path, "metric", data + 10)
    write_manifest = store.write_manifest
    sync_tree = store.sync_tree
    sync_directory = store_integrity.sync_directory

    def fail(*args, **kwargs):
        raise OSError(f"injected {stage}")

    if stage == "pack":
        monkeypatch.setattr(store, "pack", fail)
    elif stage == "payload":
        monkeypatch.setattr(zarr.Array, "__setitem__", fail)
    elif stage == "flush":
        def flush(path):
            if path.name == "metric":
                fail()
            sync_tree(path)
        monkeypatch.setattr(store, "sync_tree", flush)
    elif stage == "publish":
        def publish(path, value):
            if value["metrics"].get("metric", {}).get("state") == "complete":
                fail()
            write_manifest(path, value)
        monkeypatch.setattr(store, "write_manifest", publish)
    else:
        triggered = False
        def directory_flush(path):
            nonlocal triggered
            if not triggered and path == tmp_path and (path / "pspi-manifest.json").is_file() and \
               manifest(path)["metrics"].get("metric", {}).get("state") == "complete":
                triggered = True
                fail()
            sync_directory(path)
        monkeypatch.setattr(store_integrity, "sync_directory", directory_flush)
    with pytest.raises(OSError, match=f"injected {stage}"):
        store.write_metric(tmp_path, "metric", data, overwrite=overwrite)
    assert manifest(tmp_path)["metrics"]["metric"]["state"] != "complete"
    assert "metric" in store.verify(tmp_path)
    assert not list(tmp_path.glob(".pspi-manifest-*.tmp"))


def test_missing_chunk_after_payload_write_prevents_completion(store, tmp_path, monkeypatch):
    original = zarr.Array.__setitem__
    def discard(array, selection, values):
        original(array, selection, values)
        remove_inner(tmp_path / "metric" / "upper")
    monkeypatch.setattr(zarr.Array, "__setitem__", discard)
    with pytest.raises(store.StorageIntegrityError, match="inner chunk is missing"):
        store.write_metric(tmp_path, "metric", matrix_case("sym"))
    assert manifest(tmp_path)["metrics"]["metric"]["state"] == "writing"


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "schema", "inventory", "name", "shape", "state", "dtype", "arrays", "declarations"])
def test_invalid_manifest_is_explicit_and_never_guesses_shape(store, tmp_path, damage):
    store.write_metric(tmp_path, "metric", matrix_case("sym"))
    path = tmp_path / "pspi-manifest.json"
    value = manifest(tmp_path)
    if damage == "missing":
        path.unlink()
    elif damage == "invalid_json":
        path.write_text("{")
    else:
        entry = value["metrics"]["metric"]
        if damage == "schema":
            value["schema"] = "pspi-store/0.1"
        elif damage == "inventory":
            value["metrics"] = []
        elif damage == "name":
            value["metrics"]["../escape"] = entry
        elif damage == "shape":
            entry["meta"]["n"] = -1
        elif damage == "state":
            entry["state"] = "okay"
        elif damage == "dtype":
            entry["meta"]["dtype"] = "<i8"
        elif damage == "arrays":
            entry["meta"]["arrays"] = ["full"]
        else:
            entry["array_metadata"] = {}
        path.write_text(json.dumps(value))
    with pytest.raises((store.MetadataError, store.StorageFormatError)):
        store.open_results(tmp_path).load("metric")
    assert store.verify(tmp_path)


def test_unsupported_index_encoding_is_not_silently_recovered(store, tmp_path):
    store.write_metric(tmp_path, "metric", matrix_case("sym"))
    value = manifest(tmp_path)
    expected = value["metrics"]["metric"]["array_metadata"]["upper"]
    expected["codecs"][0]["configuration"]["index_codecs"] = [{"name": "bytes"}]
    (tmp_path / "pspi-manifest.json").write_text(json.dumps(value))
    with pytest.raises(store.StorageFormatError, match="index encoding"):
        store.open_results(tmp_path).load("metric")
    assert "metric" in store.verify(tmp_path)


@pytest.mark.parametrize("error", [TypeError, AssertionError, MemoryError])
def test_programming_errors_are_not_converted_to_nan(store, tmp_path, monkeypatch, error):
    store.write_metric(tmp_path, "metric", matrix_case("sym"))
    def broken(*args, **kwargs):
        raise error("injected programming error")
    monkeypatch.setattr(zarr.Array, "__getitem__", broken)
    with pytest.raises(error, match="injected programming error"):
        store.open_results(tmp_path).load("metric")
    with pytest.raises(error, match="injected programming error"):
        store.verify(tmp_path)


def test_partial_warning_can_be_promoted_to_error(store, tmp_path):
    store.write_metric(tmp_path, "metric", matrix_case("sym"))
    data_files(tmp_path / "metric" / "upper")[0].unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.PartialReadWarning)
        with pytest.raises(store.PartialReadWarning):
            store.open_results(tmp_path).load("metric")


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
def test_mean_uses_checked_reads_and_rejects_partial_input(store, force_mode, tmp_path, codec, kind):
    force_mode(kind)
    data = matrix_case(kind)
    store.write_metric(tmp_path, "metric", data, codec=codec)
    reader = store.open_results(tmp_path)
    assert_bits(reader.mean("metric"), np.nanmean(data, axis=2))
    key = "residual" if kind.endswith("_res") else ("full" if kind == "full" else "upper")
    data_files(tmp_path / "metric" / key)[0].unlink()
    with pytest.warns(store.PartialReadWarning):
        with pytest.raises(store.StorageIntegrityError):
            reader.mean("metric")


@pytest.mark.parametrize("stage", ["before_layout", "after_upper"])
def test_interrupted_residual_write_recovers_only_published_entries(store, force_mode, tmp_path, monkeypatch, stage):
    force_mode("sym_res")
    data = matrix_case("sym_res")
    assign = zarr.Array.__setitem__
    if stage == "before_layout":
        def stop_layout(*args, **kwargs):
            raise OSError("layout interrupted")
        monkeypatch.setattr(zarr.Group, "create_array", stop_layout)
    else:
        def stop_residual(array, selection, values):
            if array.path.endswith("/residual"):
                raise OSError("residual interrupted")
            return assign(array, selection, values)
        monkeypatch.setattr(zarr.Array, "__setitem__", stop_residual)
    with pytest.raises(OSError, match="interrupted"):
        store.write_metric(tmp_path, "metric", data)
    assert manifest(tmp_path)["metrics"]["metric"]["state"] == "writing"
    mask = np.zeros(data.shape, dtype=bool)
    if stage == "after_upper":
        i, j = np.triu_indices(4)
        mask[i, j] = True
    expect_partial(store, store.open_results(tmp_path), "metric", data, mask)
    assert "metric" in store.verify(tmp_path)


@pytest.mark.parametrize("damage", ["not_mapping", "shape", "grid", "separator", "codec", "unchecked", "geometry", "location", "missing_field"])
def test_invalid_declared_array_layout_is_an_explicit_error(store, tmp_path, damage):
    store.write_metric(tmp_path, "metric", matrix_case("sym"))
    value = manifest(tmp_path)
    spec = value["metrics"]["metric"]["array_metadata"]["upper"]
    if damage == "not_mapping":
        value["metrics"]["metric"]["array_metadata"]["upper"] = []
    elif damage == "shape":
        spec["shape"][0] += 1
    elif damage == "grid":
        spec["chunk_grid"]["name"] = "unknown"
    elif damage == "separator":
        spec["chunk_key_encoding"]["configuration"]["separator"] = "!"
    elif damage == "codec":
        spec["codecs"] = []
    elif damage == "unchecked":
        spec["codecs"][0]["configuration"]["codecs"].pop()
    elif damage == "geometry":
        spec["chunk_grid"]["configuration"]["chunk_shape"][1] = 1
    elif damage == "location":
        spec["codecs"][0]["configuration"]["index_location"] = "middle"
    else:
        del spec["chunk_grid"]
    (tmp_path / "pspi-manifest.json").write_text(json.dumps(value))
    with pytest.raises((store.MetadataError, store.StorageFormatError)):
        store.open_results(tmp_path).load("metric")
    assert "metric" in store.verify(tmp_path)


@pytest.mark.parametrize("name", ["", "../other", ".", "..", "a/b", "a\\b", "zarr.json", "pspi-manifest.json", "bad\x00name"])
def test_unsafe_metric_names_are_rejected_before_creating_output(store, tmp_path, name):
    path = tmp_path / "uncreated.pspi"
    with pytest.raises(ValueError, match="metric"):
        store.write_metric(path, name, matrix_case("sym"))
    assert not path.exists()


def test_legacy_format_is_explicitly_unsupported_and_unchanged(store, tmp_path):
    zarr.open_group(str(tmp_path), mode="w", attributes={"pspi_store": {"schema": "pspi-store/0.1"}})
    before = tree_fingerprint(tmp_path)
    with pytest.raises(store.StorageFormatError, match="manifest"):
        store.open_results(tmp_path)
    with pytest.raises(store.StorageFormatError, match="manifest"):
        store.write_metric(tmp_path, "metric", matrix_case("sym"))
    assert "__store__" in store.verify(tmp_path)
    assert tree_fingerprint(tmp_path) == before


def test_root_metadata_damage_prevents_writes_without_modifying_data(store, tmp_path):
    store.write_metric(tmp_path, "metric", matrix_case("sym"))
    (tmp_path / "zarr.json").write_text("[]")
    before = tree_fingerprint(tmp_path)
    with pytest.raises(store.MetadataError, match="root"):
        store.write_metric(tmp_path, "other", matrix_case("sym"))
    assert tree_fingerprint(tmp_path) == before


@pytest.mark.parametrize("n", [1, 4])
def test_zero_window_store_has_no_required_payload_but_valid_metadata(store, tmp_path, codec, n):
    data = np.empty((n, n, 0))
    store.write_metric(tmp_path, "metric", data, codec=codec)
    values, available = store.open_results(tmp_path).load("metric", return_availability=True)
    assert values.shape == available.shape == data.shape
    assert store.verify(tmp_path) == {}
    key = "full" if n == 1 else "upper"
    (tmp_path / "metric" / key / "zarr.json").unlink()
    assert "metric" in store.verify(tmp_path)


def test_many_missing_chunks_warn_once_and_verifier_keeps_full_details(store, tmp_path, monkeypatch):
    data = matrix_case("sym", windows=40)
    store.write_metric(tmp_path, "metric", data, target_chunk_values=20, max_shard_values=40)
    for shard in range(10):
        remove_inner(tmp_path / "metric" / "upper", shard=shard)
    monkeypatch.setattr(store, "READ_BLOCK_VALUES", 15)
    mask = np.ones(data.shape, dtype=bool)
    for start in range(0, 40, 4):
        mask[:, :, start:start + 2] = False
    message = expect_partial(store, store.open_results(tmp_path), "metric", data, mask)
    assert "additional affected ranges" in message
    assert "[36, 38)" in store.verify(tmp_path)["metric"]
