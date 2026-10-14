"""Keep the experimental storage import in one place until production integration."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def packing(monkeypatch):
    prototype = Path(__file__).resolve().parents[2] / "storage_comparison"
    monkeypatch.syspath_prepend(str(prototype))
    return importlib.import_module("packing")


@pytest.fixture
def store(packing):
    # The full pyspi environment need not install experimental storage dependencies.
    for name in ("zarr", "numcodecs", "pcodec", "zstandard", "blosc2", "zfpy", "fpzip"):
        pytest.importorskip(name, minversion="3.0" if name == "zarr" else None)
    return importlib.import_module("pspi_store")


@pytest.fixture(params=["pcodec", "zstd"])
def codec(request):
    return request.param


@pytest.fixture
def force_mode(monkeypatch, store):
    """Exercise each representation independently of compressor size heuristics."""
    def select(kind):
        monkeypatch.setattr(store, "_residual_smaller", lambda codec: lambda residual, full: kind != "full")
    return select
