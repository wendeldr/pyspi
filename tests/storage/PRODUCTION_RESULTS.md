# Production storage validation

Initial integration validated 2026-10-14 on Linux x86-64, Python 3.12.3, NumPy 2.5.4, and Java 21.0.12.1.

The counts and receipt below describe that integration checkpoint. The subsequent
[adversarial test review](../TEST_REVIEW.md) expanded storage/analysis verification
to 439 passing cases and records the latest calculation-suite counts and
coverage. The integration receipt and benchmark measurements remain unchanged.

## Installed-package verification

* **419 storage/analysis cases passed**, with no skips or expected failures, against a noneditable wheel in a fresh environment. Tests ran locally outside the checkout, importing the installed package from site-packages.
* Combined statement/branch coverage is approximately **98%**. Packing, Calculator adaptation, and `pyspi.analysis.time_mean` each have full statement and branch coverage.
* The default calculation suite retained **563 passes, 15 skips, and 10 expected failures**, with the original skip/failure identities unchanged. Original calculation functions, regression tests, goldens, and tolerance policy are unchanged.
* Both the wheel and the documented pip CPU/storage installation calculated four fabfour metrics over two 1024-sample windows. Single/windowed storage, original bits, NaN availability, labels, supplied channel names, metric order, and repeated/out-of-order selections all passed.
* Root and fresh installed dependency checks pass. The production storage extra uses only Zarr, numcodecs, pcodec, and google-crc32c. The comparison registry, lossy packing, precision conversion, and reader mean API are absent.
* Corruption/partial-write tests run against the installed package, including missing primary/residual payloads, inner chunks, metric directories, truncated/corrupt data and indexes, metadata damage, interruption and publication/flush failures. Recoverable entries, warning behavior, availability masks, and failure states are asserted.
* The original comparison source/tests are archived; reports and survey files are preserved with hashes. Generated comparison result copies were moved unchanged. Patient recordings and original patient stores were not opened or modified.

Detailed source/wheel hashes, test counts, coverage, installation provenance, and completion evidence are in [production_verification.json](production_verification.json). Raw JUnit, coverage, and console artifacts remain locally under `.artifacts/`. The unused GitHub Actions workflows were removed at the user's request; verification is local. No physical power-loss simulation is claimed.

## Bounded storage measurements

Three repetitions covered 24 combinations: 50×50×64, 118×118×96, 300×300×32, and 118×118×600; full, symmetric, and exact-residual inputs; pcodec and Zstd. Synthetic generation was excluded from timing. The largest supplied matrix was 66,835,200 bytes (63.7 MiB), with one metric held at a time; packing, encoding, and reads allocate additional working memory. Inputs were synthetic, not existing patient metrics.

Runs used the installed wheel, temporary `/tmp` stores, warm OS caches, and an AMD Ryzen Threadripper 3970X 32-Core Processor with 64 logical CPUs. Package versions, raw samples, and complete environment information are in [production_benchmark.json](production_benchmark.json). These measurements do not establish EEG calculation throughput or savings on patient data.

Median milliseconds for one 118-channel metric over 600 windows:

| Form | Codec | Packing including selection | Compression selection alone | Durable write total | Full read | Four selected windows | Verify |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | pcodec | 634.7 | 502.5 | 812.0 | 118.8 | 8.8 | 73.7 |
| full | zstd | 319.8 | 189.2 | 464.0 | 126.9 | 11.0 | 74.8 |
| sym | pcodec | 81.1 | 0.0 | 189.6 | 156.5 | 8.9 | 45.0 |
| sym | zstd | 84.6 | 0.0 | 174.3 | 160.2 | 10.2 | 46.6 |
| sym_res | pcodec | 541.4 | 424.1 | 675.3 | 175.9 | 13.9 | 70.0 |
| sym_res | zstd | 453.8 | 338.5 | 561.4 | 176.4 | 15.4 | 66.6 |

Component timers overlap: compression selection is part of packing, and packing is part of total write time. `sync_tree` and manifest-publication timers are also recorded separately in the raw JSON; the total includes their filesystem flushes. Do not sum overlapping timers.

The separate **185-metric batch of 4×4×7 arrays** took 5.70 seconds median and produced 562,752 file bytes. It exposes fixed manifest/durability overhead for tiny matrices; it is not a patient throughput estimate.

The representation-choice heuristic can dominate writes of general or near-symmetric data because it compresses alternatives before saving one. Removing the prototype's diagnostic relative-error scans shortened packing, but these runs do not establish a controlled before/after speed ratio. Stored values and the exact residual policy remain unchanged. Further selection-cost optimization and batch publication changes need their own measured scope.

## Contract and limits

The public API and installation are documented in [docs/storage.md](../../docs/storage.md). Production schema is `pspi-store/1.0`, with explicit rejection of prototype/legacy formats and no automatic migration. The implementation supports a local POSIX filesystem with one writer per recording, metric-sized in-memory packing, and selective reads. Append/resume orchestration, streaming windows, concurrent writers, object stores, and transfer repair remain future work.

Time averaging is in `pyspi.analysis`, with explicit float64 accumulation and rejection of unavailable input. The old antisymmetric subnormal signed-zero case is fixed and tested. The two prototype tolerance/infinity expected failures are archived with the unselected lossy experiment; production has no tolerance-based packing path. Scientific definition and labeling findings remain in [the archived review](../../docs/storage_comparison/REVIEW.md) for individual discussion.
