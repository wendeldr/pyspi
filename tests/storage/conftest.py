"""Exercise the installed pyspi package; no comparison-directory imports."""

import importlib

import pytest


@pytest.fixture
def packing():
    return importlib.import_module("pyspi.storage._packing")


@pytest.fixture
def store():
    return importlib.import_module("pyspi.storage._store")


@pytest.fixture(params=["pcodec", "zstd"])
def codec(request):
    return request.param


@pytest.fixture
def force_mode(monkeypatch, store):
    """Exercise each representation independently of compressor size heuristics."""
    def select(kind):
        monkeypatch.setattr(store, "_residual_smaller", lambda codec, level=None: lambda residual, full: kind != "full")
    return select
