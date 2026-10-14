"""Public reader/writer behavior using temporary stores and real codecs."""

import json
import warnings

import numpy as np
import pytest

from storage_cases import assert_bits, data_files, matrix_case, special_values, tree_fingerprint


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_every_mode_through_real_codecs(store, force_mode, codec, tmp_path, kind, dtype):
    force_mode(kind)
    data = matrix_case(kind, dtype=dtype)
    original = data.copy()
    original.flags.writeable = False
    path = tmp_path / "recording.pspi"
    meta = store.write_metric(path, "metric", original, dtype=dtype, codec=codec,
                              labels=["deliberately_unreliable_label"], target_chunk_values=20,
                              max_shard_values=60)
    assert meta["mode"] == kind
    reader = store.open_results(path)
    assert_bits(reader.load("metric"), data)
    assert_bits(reader.load("metric", windows=[6, 0, 6, 2]), data[:, :, [6, 0, 6, 2]])
    assert_bits(reader.load("metric", windows=3), data[:, :, 3])
    assert_bits(reader.load("metric", windows=[]), data[:, :, :0])
    assert_bits(original, data)
    assert reader.info("metric")["labels"] == ["deliberately_unreliable_label"]
    assert store.verify(path) == {}


@pytest.mark.parametrize("n,windows", [(1, 1), (1, 11), (2, 1), (7, 5)])
def test_small_matrices_and_window_counts(store, codec, tmp_path, n, windows):
    data = matrix_case("sym", n=n, windows=windows)
    store.write_metric(tmp_path, "small", data, codec=codec)
    assert_bits(store.open_results(tmp_path).load("small"), data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ieee_values_are_preserved_on_disk(store, codec, tmp_path, dtype):
    values = special_values(dtype)
    data = np.tile(values, 4).reshape(2, 2, -1)
    with np.errstate(invalid="ignore", over="ignore"):
        store.write_metric(tmp_path, "special", data, dtype=dtype, codec=codec)
        assert_bits(store.open_results(tmp_path).load("special"), data)
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_bit_patterns_are_preserved_on_disk(store, codec, tmp_path, dtype):
    uint = np.uint64 if dtype == np.float64 else np.uint32
    data = np.random.default_rng(9).integers(0, np.iinfo(uint).max, size=(4, 4, 5), dtype=uint).view(dtype)
    with np.errstate(invalid="ignore", over="ignore"):
        store.write_metric(tmp_path, "bits", data, dtype=dtype, codec=codec)
        assert_bits(store.open_results(tmp_path).load("bits"), data)


@pytest.mark.parametrize("value", [0.0, -0.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_invalid_or_zero_data_is_legitimate_output(store, codec, tmp_path, value, dtype):
    data = np.full((3, 3, 5), value)
    store.write_metric(tmp_path, "constant", data, codec=codec, dtype=dtype)
    assert_bits(store.open_results(tmp_path).load("constant"), data.astype(dtype))
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("layout", ["fortran", "sliced", "reversed"])
def test_writer_accepts_noncontiguous_input_without_changing_it(store, codec, tmp_path, layout):
    data = matrix_case("full", n=5, windows=8)
    if layout == "fortran":
        data = np.asfortranarray(data)
    elif layout == "sliced":
        data = data[::2, ::2, ::2]
    else:
        data = data[::-1, ::-1, ::-1]
    before = data.copy()
    store.write_metric(tmp_path, "layout", data, codec=codec)
    assert_bits(store.open_results(tmp_path).load("layout"), before)
    assert_bits(data, before)


def test_metadata_multiple_metrics_and_recording_merge(store, tmp_path):
    results = {"zeta": matrix_case("sym"), "alpha": matrix_case("anti")}
    store.write_results(tmp_path, results, labels={"alpha": ["signed", "directed"]},
                        recording={"channels": ["A", "B", "C", "D"], "sampling_rate": 2048})
    store.set_recording_info(tmp_path, {"sampling_rate": 1024, "window_onsets": [0, 512]})
    reader = store.open_results(tmp_path)
    assert reader.metrics == ["alpha", "zeta"]
    assert reader.recording == {"channels": ["A", "B", "C", "D"], "sampling_rate": 1024,
                                "window_onsets": [0, 512]}
    assert reader.info("alpha")["labels"] == ["signed", "directed"]
    assert reader.info("zeta")["labels"] == []
    for name, data in results.items():
        assert_bits(reader.load(name), data)
    info = reader.info("alpha")
    info["mode"] = "invalid_local_edit"
    assert reader.info("alpha")["mode"] == "anti"


def test_extra_metadata_and_custom_codec_level(store, tmp_path, codec):
    data = matrix_case("sym")
    store.write_metric(tmp_path, "metadata", data, codec=codec, level=4,
                       attrs={"estimator": "example", "parameters": {"window_ms": 500}})
    reader = store.open_results(tmp_path)
    assert reader.info("metadata")["parameters"] == {"window_ms": 500}
    assert reader.info("metadata")["estimator"] == "example"
    assert_bits(reader.load("metadata"), data)


def test_single_matrix_and_explicit_window_have_documented_shapes(store, tmp_path, codec):
    matrix = matrix_case("sym", windows=1)[:, :, 0]
    store.write_metric(tmp_path, "single", matrix, codec=codec)
    reader = store.open_results(tmp_path)
    assert_bits(reader.load("single"), matrix)
    assert_bits(reader.load("single", windows=0), matrix)
    assert_bits(reader.load("single", windows=[0]), matrix[:, :, None])


def test_full_reads_cross_block_and_shard_boundaries(store, tmp_path, codec, monkeypatch):
    data = matrix_case("sym", n=4, windows=19)
    monkeypatch.setattr(store, "READ_BLOCK_VALUES", 35)
    store.write_metric(tmp_path, "blocked", data, codec=codec, target_chunk_values=20,
                       max_shard_values=40)
    assert len(data_files(tmp_path / "blocked" / "upper")) > 1
    reader = store.open_results(tmp_path)
    assert_bits(reader.load("blocked"), data)
    assert_bits(reader.load("blocked", windows=[18, 3, 4, 0, 18]), data[:, :, [18, 3, 4, 0, 18]])


def test_reading_and_verification_do_not_modify_store(store, tmp_path):
    store.write_metric(tmp_path, "unchanged", matrix_case("sym"))
    before = tree_fingerprint(tmp_path)
    reader = store.open_results(tmp_path)
    reader.metrics
    reader.recording
    reader.info("unchanged")
    reader.load("unchanged")
    reader.load("unchanged", windows=[2, 0])
    assert store.verify(tmp_path) == {}
    assert tree_fingerprint(tmp_path) == before


def test_overwrite_refusal_preserves_original_data_and_metadata(store, tmp_path):
    data = matrix_case("sym")
    store.write_metric(tmp_path, "original", data, attrs={"note": "keep"})
    before = tree_fingerprint(tmp_path)
    with pytest.raises(FileExistsError):
        store.write_metric(tmp_path, "original", data + 100)
    assert tree_fingerprint(tmp_path) == before
    assert_bits(store.open_results(tmp_path).load("original"), data)


def test_explicit_overwrite_changes_only_selected_metric(store, tmp_path):
    original, neighbor = matrix_case("sym"), matrix_case("anti")
    store.write_results(tmp_path, {"chosen": original, "neighbor": neighbor})
    before = tree_fingerprint(tmp_path / "neighbor")
    store.write_metric(tmp_path, "chosen", original + 10, overwrite=True)
    assert_bits(store.open_results(tmp_path).load("chosen"), original + 10)
    assert tree_fingerprint(tmp_path / "neighbor") == before


@pytest.mark.parametrize("selection", [[7], [-8]])
def test_out_of_bounds_window_is_rejected(store, tmp_path, selection):
    store.write_metric(tmp_path, "selected", matrix_case("sym"))
    with pytest.raises(IndexError):
        store.open_results(tmp_path).load("selected", windows=selection)


def test_missing_metric_is_reported_as_missing(store, tmp_path):
    store.write_metric(tmp_path, "present", matrix_case("sym"))
    with pytest.raises(KeyError):
        store.open_results(tmp_path).load("absent")


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4), (2, 2, 2, 2)])
def test_invalid_shapes_are_rejected_before_creating_output(store, tmp_path, shape):
    path = tmp_path / "invalid.pspi"
    with pytest.raises(ValueError, match="expected"):
        store.write_metric(path, "invalid", np.ones(shape))
    assert not path.exists()


def test_unsupported_target_dtype_is_rejected(store, tmp_path):
    with pytest.raises(ValueError, match="float64 or float32"):
        store.write_metric(tmp_path / "invalid", "metric", matrix_case("sym"), dtype="int64")


def test_unknown_codec_is_rejected_instead_of_silently_using_another(store, tmp_path):
    with pytest.raises(ValueError, match="unknown codec"):
        store.write_metric(tmp_path, "metric", matrix_case("sym"), codec="unknown")


def test_float32_is_explicit_and_records_conversion(store, tmp_path):
    data = matrix_case("full") + 1 / 7
    store.write_metric(tmp_path, "default", data)
    store.write_metric(tmp_path, "rounded", data, dtype="float32")
    reader = store.open_results(tmp_path)
    assert_bits(reader.load("default"), data)
    assert_bits(reader.load("rounded"), data.astype(np.float32))
    assert reader.info("rounded")["source_dtype"] == np.dtype(np.float64).str
    assert reader.info("rounded")["float32_fallback"] is None


@pytest.mark.parametrize("value", [1e-40, 1e300, -1e300])
def test_float32_range_fallback_preserves_whole_array(store, tmp_path, value):
    data = matrix_case("sym")
    data[0, 1, 0] = value
    store.write_metric(tmp_path, "fallback", data, dtype="float32")
    reader = store.open_results(tmp_path)
    assert reader.info("fallback")["float32_fallback"]
    assert reader.info("fallback")["dtype"] == np.dtype(np.float64).str
    assert_bits(reader.load("fallback"), data)


@pytest.mark.parametrize("policy", ["error", "allow"])
def test_float32_range_policy_is_explicit(store, tmp_path, policy):
    data = matrix_case("sym")
    data[0, 1, 0] = 1e-40
    if policy == "error":
        with pytest.raises(ValueError, match="metric"):
            store.write_metric(tmp_path, "metric", data, dtype="float32", float32_out_of_range=policy)
    else:
        store.write_metric(tmp_path, "metric", data, dtype="float32", float32_out_of_range=policy)
        assert_bits(store.open_results(tmp_path).load("metric"), data.astype(np.float32))


def test_float32_normal_boundaries_and_existing_infinities_are_supported(store, tmp_path):
    info = np.finfo(np.float32)
    data = np.array([info.tiny, info.max, -info.max, np.inf, -np.inf, np.nan, 0, -0.0]).reshape(2, 2, 2)
    assert store.float32_range_problem(data) is None
    store.write_metric(tmp_path, "range", data, dtype="float32")
    assert_bits(store.open_results(tmp_path).load("range"), data.astype(np.float32))


@pytest.mark.parametrize("selection", [
    1.9, 1.0, True, False, np.bool_(True), np.float32(1),
    [0, 1.5], [0, 1.0], [True, False, True], [0, True], [False, 1],
    np.array([True, False]), np.array([0, True], dtype=object),
    np.array([0.0, 1.0]), "1", ["1"], [None], [1 + 0j],
    np.datetime64(1, "ns"), np.array([1], dtype="datetime64[ns]"),
    np.timedelta64(1, "ns"), np.array([1], dtype="timedelta64[ns]"),
])
def test_ambiguous_window_indices_are_not_silently_coerced(store, tmp_path, selection):
    store.write_metric(tmp_path, "selected", matrix_case("sym"))
    with pytest.raises(TypeError, match="integer indices"):
        store.open_results(tmp_path).load("selected", windows=selection)


@pytest.mark.parametrize("selection", [
    0, -1, -7, np.int32(3), np.uint64(3), np.array(3),
    [0, -1, 0, -7], (6, 1, 6), range(0, 7, 2),
    np.array([6, 0, 6], dtype=np.int8), np.array([6, 0], dtype=np.uint64),
    np.array([0, -1], dtype=object), [], np.array([], dtype=np.int64),
])
def test_integer_selections_preserve_shape_order_duplicates_and_availability(store, tmp_path, selection):
    data = matrix_case("sym")
    store.write_metric(tmp_path, "selected", data)
    values, available = store.open_results(tmp_path).load("selected", selection, return_availability=True)
    index = int(selection) if np.ndim(selection) == 0 else np.asarray(selection, dtype=np.int64)
    assert_bits(values, data[:, :, index])
    assert available.shape == values.shape and available.all()


@pytest.mark.parametrize("selection", [
    2**80, -(2**80), [2**80], [-(2**80)],
    np.uint64(2**64 - 1), np.array([2**64 - 1], dtype=np.uint64),
])
def test_large_integer_indices_never_wrap_to_valid_windows(store, tmp_path, selection):
    store.write_metric(tmp_path, "selected", matrix_case("sym"))
    with pytest.raises(IndexError, match="outside"):
        store.open_results(tmp_path).load("selected", selection)


@pytest.mark.parametrize("selection", [[[0, 1]], np.zeros((0, 2), dtype=np.int64)])
def test_multidimensional_window_selections_are_rejected(store, tmp_path, selection):
    store.write_metric(tmp_path, "selected", matrix_case("sym"))
    with pytest.raises(IndexError, match="one-dimensional"):
        store.open_results(tmp_path).load("selected", selection)


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_oversized_windows_warn_and_round_trip_in_one_window_shards(store, force_mode, tmp_path, codec, kind, dtype):
    force_mode(kind)
    data = matrix_case(kind, windows=5, dtype=dtype)
    with pytest.warns(store.ShardSizeWarning, match="one window per shard") as caught:
        meta = store.write_metric(tmp_path, "wide", data, codec=codec, dtype=dtype, max_shard_values=5)
    assert meta["mode"] == kind
    assert len(caught) == len(meta["arrays"])
    for key in meta["arrays"]:
        array_path = tmp_path / "wide" / key
        spec = json.loads((array_path / "zarr.json").read_text())
        shard_shape = spec["chunk_grid"]["configuration"]["chunk_shape"]
        inner_shape = spec["codecs"][0]["configuration"]["chunk_shape"]
        assert shard_shape[0] == inner_shape[0] == 1
        assert shard_shape[1] > 5
        assert len(data_files(array_path)) == data.shape[2]
        message = next(str(w.message) for w in caught if f"wide/{key}:" in str(w.message))
        assert f"{shard_shape[1]} stored values" in message
        assert f"{shard_shape[1] * np.dtype(dtype).itemsize} uncompressed bytes" in message
        assert "target of 5 values" in message
    values, available = store.open_results(tmp_path).load("wide", return_availability=True)
    assert_bits(values, data)
    assert available.all()
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("target", [16, 35])
def test_shard_target_reduces_inner_chunks_when_each_window_fits(store, force_mode, tmp_path, codec, kind, target):
    force_mode(kind)
    data = matrix_case(kind, windows=11)
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.ShardSizeWarning)
        meta = store.write_metric(tmp_path, "fits", data, codec=codec,
                                  target_chunk_values=1000, max_shard_values=target)
    for key in meta["arrays"]:
        spec = json.loads((tmp_path / "fits" / key / "zarr.json").read_text())
        assert np.prod(spec["chunk_grid"]["configuration"]["chunk_shape"]) <= target
        assert np.prod(spec["codecs"][0]["configuration"]["chunk_shape"]) <= target
    assert_bits(store.open_results(tmp_path).load("fits"), data)
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("windows", [0, 1, 5])
def test_target_boundary_and_empty_payload_do_not_emit_size_warning(store, tmp_path, codec, windows):
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.ShardSizeWarning)
        exact = np.ones((5, 5, windows))  # 15 upper values: exactly the target
        store.write_metric(tmp_path, "boundary", exact, codec=codec, max_shard_values=15)
        if not windows:
            store.write_metric(tmp_path, "empty", exact, codec=codec, max_shard_values=1)
    assert_bits(store.open_results(tmp_path).load("boundary"), exact)
    assert store.verify(tmp_path) == {}


@pytest.mark.parametrize("name", ["target_chunk_values", "max_shard_values"])
@pytest.mark.parametrize("value,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
@pytest.mark.parametrize("batch", [False, True])
def test_invalid_size_targets_are_rejected_before_modifying_output(store, tmp_path, name, value, error, batch):
    data = matrix_case("sym")
    store.write_metric(tmp_path, "original", data)
    before = tree_fingerprint(tmp_path)
    options = {name: value, "overwrite": True}
    with pytest.raises(error, match=name):
        if batch:
            store.write_results(tmp_path, {"original": data + 10}, **options)
        else:
            store.write_metric(tmp_path, "original", data + 10, **options)
    assert tree_fingerprint(tmp_path) == before


def test_numpy_integer_size_targets_are_supported(store, tmp_path):
    data = matrix_case("sym")
    store.write_metric(tmp_path, "sized", data, target_chunk_values=np.int64(20), max_shard_values=np.uint64(40))
    assert_bits(store.open_results(tmp_path).load("sized"), data)
    assert store.verify(tmp_path) == {}


def test_size_warning_as_error_never_marks_an_incomplete_write_complete(store, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", store.ShardSizeWarning)
        with pytest.raises(store.ShardSizeWarning):
            store.write_metric(tmp_path, "wide", matrix_case("sym"), max_shard_values=5)
    record = json.loads((tmp_path / "pspi-manifest.json").read_text())
    assert record["metrics"]["wide"]["state"] == "writing"
    assert "wide" in store.verify(tmp_path)
