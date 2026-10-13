"""Access examples: how much code a downstream user needs for the common tasks.

Writes a few real metrics of one patient (read-only source) into the prototype store and a
Parquet copy, then shows: (1) the pspi_store API, (2) plain zarr + numpy without pspi_store,
(3) one SQL statement in DuckDB over Parquet. Each path is checked against the source mean.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

from pspi_store import open_results, write_metric

SOURCE = Path("/media/dan/Data/outputs/calculated_seeg_patient_data")


def main():
    patient = sys.argv[1] if len(sys.argv) > 1 else "105"
    metrics = ["cov_EmpiricalCovariance", "wpli_multitaper_mean_fs-1_fmin-0_fmax-0-5", "te_symbolic_k-1_kt-1_l-1_lt-1"]
    tmp = Path(tempfile.mkdtemp(prefix="pspi_access_"))
    try:
        store = tmp / f"{patient}.pspi"
        sources = {}
        for m in metrics:
            src = zarr.open_array(str(SOURCE / f"{patient}.zarr/connectivity/0.5s/{m}"), mode="r")
            data = src[:]
            sources[m] = data
            write_metric(store, m, data, labels=src.attrs.get("pyspi_labels", []))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = {m: np.nanmean(d, axis=2) for m, d in sources.items()}

        # (1) pspi_store API: two lines
        r = open_results(store)
        for m in metrics:
            t = time.perf_counter()
            mean = r.mean(m)
            print(f"[api]   {m[:40]:40s} mode={r.info(m)['mode']:8s} mean {mean.shape} in {time.perf_counter() - t:.3f}s",
                  "bit-exact" if mean.tobytes() == expected[m].tobytes() else "MISMATCH")

        # (2) plain zarr + numpy, no pspi_store (symmetric metric)
        g = zarr.open_group(str(store / metrics[0]), mode="r")
        n = g.attrs["pspi"]["n"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tri = np.nanmean(g["upper"][:], axis=0)
        mean = np.zeros((n, n))
        iu = np.triu_indices(n)
        mean[iu] = tri
        mean[iu[1], iu[0]] = tri
        print(f"[zarr]  {metrics[0]} plain-numpy mean max |diff| = {np.nanmax(np.abs(mean - expected[metrics[0]])):.2e}"
              " (summation order differs from axis=2 nanmean)")

        # (3) DuckDB SQL over a Parquet export (full rows for clarity)
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq
        d = sources[metrics[2]]
        nn, _, w = d.shape
        i, j = np.unravel_index(np.arange(nn * nn), (nn, nn))
        pq.write_table(pa.table({"window": np.repeat(np.arange(w, dtype=np.uint16), nn * nn),
                                 "i": np.tile(i.astype(np.uint16), w), "j": np.tile(j.astype(np.uint16), w),
                                 "value": np.moveaxis(d, 2, 0).ravel()}),
                       tmp / "te.parquet", compression="zstd", use_byte_stream_split=["value"])
        t = time.perf_counter()
        q = duckdb.sql(f"SELECT i, j, avg(value) FILTER (WHERE NOT isnan(value)) AS m FROM '{tmp / 'te.parquet'}' "
                       "GROUP BY i, j ORDER BY i, j").fetchnumpy()
        sql_mean = q["m"].astype(float).reshape(nn, nn)
        print(f"[duckdb] {metrics[2]} SQL mean in {time.perf_counter() - t:.3f}s; max |diff| = "
              f"{np.nanmax(np.abs(sql_mean - expected[metrics[2]])):.2e}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
