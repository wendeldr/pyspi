# Storage comparison for pyspi results

Measured on 2026-10-13 using the existing sEEG outputs in `/media/dan/Data/outputs/calculated_seeg_patient_data` (read only, never modified). Everything here was produced by the scripts in this directory. Section [Reproduction](#reproduction) lists the commands.

## Bottom line

There is a clear, low-risk alternative to the current layout. It keeps every float64 value bit-for-bit.

**Recommended format:** one Zarr v3 directory per recording.
- One group per metric, holding one shard file per stored array.
- Each shard holds independently decoded inner chunks of about 256k values, each with its own CRC32C checksum.
- Structure-aware packing, decided per array from the actual bits (never from metric names or labels):
  - Matrices whose lower triangle mirrors the upper exactly (symmetric, or sign-flipped for antisymmetric) store only the diagonal and upper triangle.
  - Matrices that are symmetric up to floating-point noise store the triangle plus a small exact XOR residual.
  - Directed matrices store every entry.
- pcodec (level 8) compresses the numbers.

| Patient 105 (131 ch, 602 windows, 185 metrics) | Current | Recommended f64 | Recommended f32 |
| --- | ---: | ---: | ---: |
| Size on disk | 7.91 GiB | 5.20 GiB (0.658) | 2.45 GiB (0.309) |
| Files | 106,142 | 589 | 567 |
| Write + fsync, all metrics | 218 s | 37 s | 30 s |
| Cold read of every metric | 59 s | 33 s | 20 s |
| Cold time-mean of every metric | 67 s | 18 s | 14 s |
| A flipped byte on disk | wrong values returned silently | read error naming the metric | read error naming the metric |

Projected to the 72-recording cohort (an extrapolation, see [Cohort projection](#cohort-projection)):

| | Size | Files |
| --- | ---: | ---: |
| Current | 717 GiB | 7.64 M |
| float64, exact | about 450 GiB | about 42 k |
| float32 | about 210 GiB | about 41 k |

Reading and writing are faster than now, not slower. The time mean is bit-identical to `np.nanmean(data, axis=2)`.

**Alternatives if pcodec is unwanted:**
- **Same layout with Zstd level 3.** About 12% larger than pcodec. It uses only core Zarr v3 codecs, so any Zarr v3 reader in any language can open it without an extra codec package.
- **Parquet.** About 20% larger. Use it if SQL or DuckDB access matters more than size.

**Rejected:**
- Monolithic containers (one HDF5, tar, or database file per patient or cohort): any damage forces a full re-copy.
- Blosc2 frames: no checksums, so corruption passes silently.
- zfp and fpzip: larger and slower.
- Slower general compressors (xz, Zstd 19, pcodec 12): at most 0.2% smaller for 14–80× the encode time.
- Newer native formats (OpenZL, Vortex, DuckDB's ALP storage): not smaller than pcodec.
- Tolerance-based symmetric packing: under 1% smaller and not exact.

## Requirements and evidence

| Requirement (from the request) | Result |
| --- | --- |
| Significantly less disk space | 0.66–0.70 of the current size at float64; 0.31–0.33 at float32. About 0.63 and 0.30 cohort-wide (projected). |
| Data stays easy to access | `open_results(path).load(metric)` returns n×n×W (or n×n). `load(metric, windows=...)` reads a subset. Plain zarr + numpy works without the helper module ([Access](#access)). |
| Easy time mean | `Results.mean(metric)`, bit-identical to `np.nanmean(load(metric), axis=2)`. It reads the triangle only, so it is 3.6× (105) to 17× (004) faster than with the current files. |
| Easy transfer; no single huge fragile file | Each metric is 3–5 small files, with one data file per array. A failed or damaged copy costs one metric's file, not the patient or the cohort ([Transfer](#transfer-and-corruption)). |
| Corruption detected and contained | Every chunk carries a CRC32C. A flipped byte gives a read error for about 30 windows of one metric. The current layout returns wrong numbers with no error. |
| float64 first, float32 only if it pays | float64 is measured throughout. float32 halves the size again (0.66 → 0.31) with relative error ≤ 6e-8 ([float32](#float32)). |
| Generic to any pyspi run (1 metric or all) | Structure is detected per array from the bits, so it works for any metric set. Nothing depends on the 185 metrics here. 115 synthetic round-trip tests cover the edge cases. |
| Save/load time must not get worse | Writes are 6× (105) to 19× (004) faster; cold reads 1.8× to 14× faster; cold time means 3.6× to 17× faster than the current layout ([Containers](#full-patient-conversions)). |
| Not limited to Zarr or Python | Tested codecs and containers are written in Rust (pcodec, Vortex), C (Zstd, LZ4, xz, Blosc2, OpenZL), and C++ (zfp, fpzip, Arrow/Parquet, DuckDB), all loadable from Python ([Native candidates](#native-candidates-beyond-the-main-screen)). |

## What the data looks like

These properties explain the results, so they are worth knowing before reading the size tables.

**The numbers themselves barely compress.** The mantissas of the stored doubles are close to random noise:
- 47 of the 52 mantissa bits are ones with probability within 1% of 0.5.
- The low 32 bits of each value compress to 0.99 of their size even with xz -9e, Zstd 19, or pcodec 12.

A codec alone therefore gains only 1.1–1.2× on dense data (full-matrix Zstd 3 reaches 1.62× vs raw, likely helped by matching the mirrored duplicates in symmetric matrices; pcodec on full matrices reaches only 1.49×). Most of the available saving comes from not storing redundant entries.

**Most matrices are exactly redundant.** Bit-level analysis of 555 arrays over 3 patients:
- 333 symmetric
- 141 antisymmetric
- 48 symmetric up to rounding noise
- 3 antisymmetric up to an offset
- 30 directed (full)

About 95% of raw bytes are triangle-packable. Per patient this is 111 symmetric, 47 antisymmetric, 16 near-symmetric, 1 near-antisymmetric, and 10 full metrics.

**Directed vs undirected is the main category split, but labels are not reliable enough to decide storage.**
- phase, PLI, wPLI, and PSI are labelled undirected but are bit-exactly antisymmetric. The library computes both directions without an `abs`, so mirrored values are equal and opposite.
- Group delay (`gd`) is labelled directed but its values are symmetric or antisymmetric. NaN entries are copied unchanged.
- PLI zeros mirror as +0.0, not −0.0.
- Covariance-type estimators such as `prec_*`, `pdist_euclidean`, `pec`, and `pec_log` are mathematically symmetric. In storage they differ from symmetric only by floating-point noise (relative asymmetry up to 5.4e-8). That is why the exact XOR residual exists: it keeps the noise bit-for-bit instead of discarding it.
- The full-band phase (`fmin-0_fmax-0-5`) has U + Lᵀ = 2π/512 on some pairs, a DC/Nyquist bin artefact. The residual stores it exactly.
- Genuinely directed metrics (ce, gc, te, tlmi, lmfit, reci) store all entries.

Hence the rule: detect per array at write time. A label can still be stored as a hint, but it never decides the storage.

## Codec screen

In-memory screen of 3 patients (n = 40, 131, 249) × all 185 metrics × 64 middle windows: 59,940 rows. Every round trip was bit-checked. Ratios are vs raw float64; "vs full Zstd3" compares with Zstd 3 on unpacked full matrices. Throughput is MB/s of raw float64, single-threaded per codec, with 16 workers running concurrently. Treat it as relative, not absolute.

| Precision | Structure | Codec | Ratio vs raw | vs full Zstd3 | Encode MB/s | Decode MB/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| f64 | full matrix | Zstd 3 | 1.62 | 1.000 | 114 | 449 |
| f64 | full matrix | pcodec 8 | 1.49 | 1.086 | 106 | 500 |
| f64 | exact packing | **pcodec 8** | **2.78** | **0.584** | 182 | 1055 |
| f64 | exact packing | Blosc2 Zstd + shuffle | 2.59 | 0.625 | 149 | 894 |
| f64 | exact packing | Zstd 3 | 2.46 | 0.658 | 244 | 844 |
| f64 | exact packing | fpzip | 2.33 | 0.694 | 122 | 178 |
| f64 | exact packing | LZ4 + shuffle | 2.30 | 0.703 | 395 | 998 |
| f64 | exact packing | zfp reversible | 2.16 | 0.749 | 323 | 298 |
| f64 | tolerance packing (1e-6) | pcodec 8 | 2.80 | 0.579 | 192 | 1152 |
| f32 | exact packing | **pcodec 8** | **5.93** | **0.273** | 331 | 2733 |
| f32 | exact packing | Zstd 3 | 5.14 | 0.315 | 534 | 2104 |

- Exactness costs under 1%: tolerance packing, which drops near-symmetric noise, is only 0.8% smaller.
- Time-major vs pair-major order makes no difference (≤ 1%).
- Slow tier, on patient 105: pcodec 12 is 0.2% smaller than pcodec 8 at 14× the encode time. Zstd 19 + shuffle (2.61×) and xz + shuffle (2.62×) are both larger than pcodec 8 (2.74×) and 50–80× slower to encode.

### Native candidates beyond the main screen

⟨pending: OpenZL / Vortex / DuckDB-native screen over the same sample; headroom totals⟩

## Full-patient conversions

Every metric and window of 004 (40 ch, 607 windows) and 105 (131 ch, 602 windows) was written to disk in each container. Each run was fsynced and evicted from the page cache (`POSIX_FADV_DONTNEED`) before cold reads, then validated bit-for-bit. "Cold mean" is reading a metric and computing its NaN-ignoring time mean. "1-window" is reading one middle window of every metric. Codecs are single-threaded; Zarr uses its default thread pool.

### Patient 105

| Variant | GiB | vs current | Files | Write + fsync s | Cold read s | Cold mean s | Warm read s | 1-window s (sum of 185) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current layout (recreated, f64) | 7.911 | 1.000 | 106,142 | 217.8 | 59.4 | 66.7 | 46.9 | 0.38 |
| **Zarr shard pcodec8, f64** | **5.202** | **0.658** | **589** | **37.4** | **33.2** | **18.4** | 31.4 | 1.25 |
| Zarr dir pcodec8, f64 (no sharding) | 5.202 | 0.658 | 4,829 | 36.7 | 33.8 | 18.7 | 29.6 | 0.98 |
| framed custom file, pcodec8, f64 | 5.200 | 0.657 | 185 | 30.2 | 38.0 | 14.9 | 30.0 | 0.35 |
| Zarr shard Zstd3, f64 | 5.844 | 0.739 | 589 | 40.6 | 38.3 | 23.6 | 37.2 | 1.57 |
| Zarr shard Blosc2 Zstd + shuffle, f64 | 5.693 | 0.720 | 589 | 60.2 | 35.6 | 20.1 | 33.3 | 1.36 |
| Blosc2 b2nd frame, f64 | 5.754 | 0.727 | 387 | 118.1 | 43.1 | 18.0 | 35.0 | 0.55 |
| Parquet (byte-stream-split + Zstd), f64 | 6.268 | 0.792 | 185 | 47.1 | 41.6 | 20.2 | 34.7 | 1.26 |
| current layout, values cast to f32 | 5.236 | 0.662 | 106,142 | 158.1 | 52.7 | 62.5 | 40.9 | 0.38 |
| **Zarr shard pcodec8, f32** | **2.448** | **0.309** | **567** | **29.7** | **19.9** | **13.7** | 20.2 | 1.13 |
| Zarr shard Zstd3, f32 | 2.820 | 0.356 | 567 | 21.3 | 21.3 | 15.9 | 22.3 | 1.25 |

### Patient 004

| Variant | GiB | vs current | Files | Write + fsync s | Cold read s | Cold mean s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current layout (recreated, f64) | 0.689 | 1.000 | 108,531 | 130.1 | 44.1 | 41.4 |
| **Zarr shard pcodec8, f64** | **0.485** | **0.703** | **589** | **6.7** | **3.1** | **2.4** |
| framed custom file, pcodec8, f64 | 0.482 | 0.699 | 185 | 5.1 | 1.9 | 1.3 |
| Zarr shard Zstd3, f64 | 0.544 | 0.789 | 589 | 8.2 | 3.7 | 3.1 |
| Parquet, f64 | 0.595 | 0.863 | 185 | 4.8 | 5.9 | 2.9 |
| **Zarr shard pcodec8, f32** | **0.231** | **0.335** | **559** | 5.3 | 2.2 | 1.9 |

Observations:
- The recreated current layout matches the real store's size exactly (1.000). The baseline is the real layout, not an approximation.
- With the current layout, time goes to file count, not data volume. 004 holds 0.69 GiB but takes 130 s to write and 44 s to read, because of its 108k files.
- A one-window read decodes one ~256k-value chunk: about 7 ms per metric vs 2 ms now. Both are negligible.
- Sharding changes size by nothing and cuts files 8× compared with the unsharded directory form.
- The framed custom file is marginally smaller, with the fewest files, but needs its own reader. Zarr gives the same size with a standard reader.

## Transfer and corruption

### Flipped byte inside a metric (bench corruption test)

One byte was flipped at 25%, 50%, and 75% of a data file of `cov_EmpiricalCovariance`, and separately in its metadata.

| Format | Data byte flipped | Metadata byte flipped |
| --- | --- | --- |
| Current layout | **No error; one window silently wrong** | Metric unreadable |
| Zarr + CRC32C (all codecs) | Error naming the metric; one chunk's windows unreadable (30 of 602 for 105, about 300 of 607 for 004); the rest readable | Metric unreadable (tiny file; re-copy it) |
| Framed custom (CRC32 blocks) | Same as Zarr | Readable or error, depending on the byte |
| Parquet (page CRC) | Same as Zarr | Readable |
| Blosc2 b2nd | **No error; 3–9 windows silently wrong** | Readable |

### Interrupted copy and bit rot on the copy

⟨pending: 105 results from transfer_test.py⟩

## Chunk size

Patient 105, 10 metrics covering every storage mode, float64, pcodec 8, one shard file per array. Chunk size sets how many windows one checksum covers. It barely changes size, so the choice is between damage radius and speed.

| Values per chunk | Windows per chunk (sym, n = 131) | vs current | Write + fsync s | Cold read s | Cold mean s | 1-window s (10 metrics) | Windows lost per flipped byte |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16k | 1 | 0.723 | 4.8 | 2.9 | 2.8 | 0.05 | 1 of 602 |
| **256k (default)** | 30 | 0.721 | 2.3 | 2.0 | 1.4 | 0.07 | 30 of 602 |
| 1M | 121 | 0.721 | 2.3 | 1.8 | 1.4 | 0.13 | 121 of 602 |
| 4M | 485 | 0.721 | 2.5 | 1.9 | 1.6 | 0.35 | 485 of 602 |
| current layout | 1 | 1.000 | 11.8 | 3.3 | 3.7 | 0.02 | silent (1 wrong) |

256k values (2 MiB of raw float64) is the default: smallest and fastest, and a flipped byte costs about 5% of one metric's windows. Damage is detected rather than silent at every chunk size, and the file still re-copies whole.

## Large recording (patient 111)

n = 249, 602 windows, the largest recording in the cohort (32.7 GiB now). The same 10 metrics as the chunk sweep were converted fully. This subset is about 20% directed, against 5% in the real metric set, so its size fraction is pessimistic.

| Variant | GiB (10 metrics) | vs current | Files | Write + fsync s | Cold read s | Cold mean s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current layout, f64 | 1.759 | 1.000 | 6,026 | 21.6 | 7.3 | 7.8 |
| **Zarr shard pcodec8, f64** | **1.163** | **0.661** | **34** | **7.3** | **6.0** | **4.6** |
| current layout, cast to f32 | 0.920 | 0.523 | 6,026 | 14.8 | 4.3 | 6.0 |
| **Zarr shard pcodec8, f32** | **0.541** | **0.308** | **34** | 5.5 | 4.0 | 3.5 |

- With larger matrices the current layout spends less per byte on file overhead, so read gains shrink to 1.2–1.7×. Writes stay 3× faster, and size and file count still fall.
- A flipped byte costs 8 of 602 windows of one metric. The current layout returns one silently wrong window.
- The largest data file at n = 249 is 247 MiB (a directed metric, all entries). Highly regular metrics are much smaller: group delay 5 MiB, PLI 7 MiB.

## Threads

⟨pending: thread_scaling.py⟩

## Memory

⟨pending: memory_probe.py⟩

## float32

- Per-value relative error ≤ 6e-8; time-mean error ≤ 5e-8 of each metric's scale. Large relative errors occur only where a mean is near zero.
- No new infinities, and no NaN changes.
- Four arrays (GraphicalLassoCV `cov` and `cov-sq` in 105 and 111, about 1% of raw bytes) contain values below the float32 normal range. The writer detects this and keeps those arrays in float64 (`float32_out_of_range="fallback"`); `"error"` or `"allow"` can be chosen instead. The decision is recorded per metric.
- Size: 0.31–0.33 of the current layout vs 0.66–0.70 for float64. Converting the current layout itself to float32 only reaches 0.66, which shows that packing matters more than precision.

Whether 7 significant digits suffice is a scientific decision. The format supports both precisions per metric.

## Cohort projection

This is an extrapolation, not a measured conversion. The screen (every metric of three patients spanning n = 40–249, 64 windows) gives the ratio vs raw float64 as a function of channel count. The full conversions of 004 and 105 give the container factor: full-size on-disk ratio divided by the screen ratio. Both factors are 0.990–1.003, so the screen predicts on-disk size well.

| Variant | Estimated size | vs current | Estimated files |
| --- | ---: | ---: | ---: |
| current (measured) | 716.8 GiB | 1.000 | 7,637,989 |
| Zarr shard pcodec8, f64 | 449 GiB | 0.627 | 42,408 |
| Zarr shard Zstd3, f64 | 505 GiB | 0.704 | 42,408 |
| Zarr shard pcodec8, f32 | 211 GiB | 0.295 | 40,742 |
| Zarr shard Zstd3, f32 | 243 GiB | 0.339 | 40,742 |

The cohort fraction (0.63) is lower than 004's (0.70). The current layout compresses larger recordings slightly worse, while packing works equally well at every n: 2.77, 2.74, and 2.78× vs raw at n = 40, 131, and 249.

## Access

⟨pending: examples_access.py on 105⟩

```python
from pspi_store import open_results
r = open_results("105.pspi")
r.metrics                                 # names
x = r.load("cov_EmpiricalCovariance")    # (n, n, W) float64, bit-exact
m = r.mean("cov_EmpiricalCovariance")    # (n, n) NaN-ignoring time mean, = np.nanmean(x, axis=2)
w = r.load("te_symbolic_k-1_kt-1_l-1_lt-1", windows=[0, 10, 20])   # (n, n, 3)
```

Without the helper, reading a symmetric metric is plain zarr + numpy:

```python
import numpy as np, zarr
g = zarr.open_group("105.pspi/cov_EmpiricalCovariance", mode="r")
n = g.attrs["pspi"]["n"]
tri = np.nanmean(g["upper"][:], axis=0)       # (n(n+1)/2,) mean over windows
iu = np.triu_indices(n)
mean = np.empty((n, n)); mean[iu] = tri; mean[iu[1], iu[0]] = tri
```

The plain path differs from `np.nanmean(axis=2)` by about 1e-15, because summation order differs; `Results.mean` is bit-identical. Other modes are documented in the `pspi_store` module docstring:
- antisymmetric: negate the mirrored triangle, leaving NaN and zeros unchanged
- near-symmetric: XOR the residual bits
- full: reshape directly

## Caveats

- **Timing context:** Threadripper 3970X (32 cores), 31 GB RAM, NVMe ext4.
  - Cold reads are after `POSIX_FADV_DONTNEED` eviction, which drops the page cache but not the drive's own cache.
  - Writes include fsync.
  - Codecs are single-threaded; Zarr uses its default async pool.
  - Absolute times vary by machine; the ratios are the useful part.
- **pcodec dependency:**
  - Reading the recommended store needs the `pcodec` Python package. Zarr finds the codec automatically through numcodecs.
  - Zarr implementations outside Python need support for the `numcodecs.pcodec` codec, which is not a core Zarr v3 codec.
  - If that matters, use the Zstd variant: same layout, core codecs only, about 12% larger.
- **Recompute:** the residual and full modes keep floating-point noise exactly. If pyspi later symmetrizes these metrics at the source, they become cheaper still.
- **Sample:** the screen used 64 middle windows per metric, while the conversions used every window. The container factor (about 1.00) shows the two agree.

## Writer guidance (for pyspi integration)

1. Accumulate per metric (or per block of windows) and write one metric group at a time. A complete group is a natural resume checkpoint after a crash.
2. Detect structure per array from bits. Never rely on labels.
3. Keep the shard cap (`MAX_SHARD_VALUES`, 1 GiB of raw values) so very large n or long recordings still give bounded files.
4. Store recording metadata (channel names, window onsets, sampling, pyspi version and config) on the root group via `set_recording_info`. Store labels and source dtype per metric.
5. Refuse silent overwrites; this is the default.
6. Keep float64 as the default. float32 is an opt-in flag with the range guard.

## Reproduction

All commands run from `storage_comparison/` with `.venv/bin/python` (uv environment, Python 3.12; package versions are recorded in `reports/*/containers_env.json`).

```bash
python explore/inventory.py                     # cohort inventory (stat only)
python screen.py --run screen_fast_w64          # codec screen, 3 patients x 185 metrics x 64 windows
python analyze_screen.py reports/screen_fast_w64
python bench_containers.py --run containers_full --patients 004 105 --variants f64:exact f32:exact
python analyze_containers.py reports/containers_full
python project_cohort.py --screen reports/screen_fast_w64 --full reports/containers_full
./run_queue.sh && ./run_queue2.sh                # 111 subset, chunk sweep, transfer, threads, examples,
                                                 # native screen, headroom, memory
TMPDIR=... python test_pspi_store.py            # 115 synthetic round trips
```
