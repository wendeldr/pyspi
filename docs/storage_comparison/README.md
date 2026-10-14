# Archived storage comparison

The production implementation is now [pyspi.storage](../storage.md). This
directory preserves the original comparison's reports, scientific findings,
survey outputs, and useful timing evidence. Its Python implementations and
experimental codecs are archived, with no active production or test imports.

* [REPORT.md](REPORT.md): original storage comparison and workload measurements.
* [REVIEW.md](REVIEW.md): storage correctness review and unresolved scientific
  questions, including PSI, signed PLI/wPLI, phase aggregation, and labels.
* [PLAN.md](PLAN.md): historical comparison scope.
* `reports/` and `explore/`: preserved measurements and matrix surveys. Files are
  checked against [evidence_manifest.json](evidence_manifest.json). Large raw
  rows and console logs remain local/ignored, as in the original comparison;
  tracked summaries remain available in Git.
* [source-a2c3300.tar.gz](source-a2c3300.tar.gz): original tracked comparison
  source, experimental registry, environments, and storage tests from commit
  `a2c3300`, including the old integrity benchmark. Report text is retained here;
  only relative test-document links were adjusted for the move.

The generated `runs/` directory was moved here unchanged rather than deleting
its result copies. It remains ignored. Original patient recordings and output
stores were not opened or modified. The old derived comparison virtual
environment and bytecode caches were removed.

To inspect historical code, extract the archive into a separate scratch
directory. For historical integrity timing reproduction, that extraction
contains `storage_comparison/pyproject.toml`, its lock, and
`tests/storage/benchmark_integrity.py`; the older baseline writer can be obtained
with `git show ae55c26949d1144337e804a5ec1aaf7dd432cda3:storage_comparison/pspi_store.py`.
Use the archived comparison environment, not the production storage extra.
This is historical code with documented defects, not a supported storage API.

Production tests and bounded timings are described in
[PRODUCTION_RESULTS.md](../../tests/storage/PRODUCTION_RESULTS.md).
No mathematical definition or label finding was resolved as part of storage
integration. Those decisions remain individual follow-up discussions.
