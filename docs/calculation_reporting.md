# Calculator completion reporting and timing

`Calculator.compute()` still returns `None`, writes the usual results table,
sets matrix diagonals to NaN, warns on metric exceptions, and continues to the
next metric after those exceptions. Preprocessing flags and metric definitions
are unchanged.

Every finished metric attempt now publishes a `MetricReport` in the read-only
`calc.metric_reports` mapping. An optional callback receives that report and
the completed matrix immediately, before the next metric starts:

```python
import json
import numpy as np
from pyspi.calculator import Calculator

calc = Calculator(dataset=np.random.default_rng(42).normal(size=(3, 128)),
                  subset="sonnet", detrend=True, normalise=True)

def completed(report, values):
    print(json.dumps(report.as_dict()))
    if report.status == "success":
        assert values.shape == (3, 3)
        # The EEG runner buffers this matrix in its current bounded batch.
    else:
        assert values is None
        print(report.error.type, report.error.message)

calc.compute(on_metric_complete=completed)
reports = calc.metric_reports
```

`MetricReport`, `MetricWarning`, and `MetricError` are immutable dataclasses in
`pyspi.calculation`. Reports contain diagnostic fields, not result arrays or
live exception objects. `report.as_dict()` returns an independent representation
that Python's standard JSON encoder accepts.

## Status, values, and ordering

- `status="success"` means the metric returned and Calculator applied its
  diagonal masking and stored the matrix successfully. All-NaN, partially NaN,
  and infinite results can be successful. This status does not establish
  mathematical correctness or convergence beyond what the estimator reports.
  Errors swallowed internally by an SPI and converted to NaNs remain successful
  at this boundary; reporting does not rewrite those implementations.
- `status="failed"` means an exception occurred during that attempt, including
  diagonal masking or table assignment. `error` describes the fully qualified
  exception type, message, and formatted traceback, including chained causes.
  The legacy table block is filled with NaNs; the callback receives `None`.
- A successful callback receives a detached NumPy copy of the stored matrix,
  preserving dtype, bit patterns, axis order, and the masked diagonal. The copy
  is initially read-only and remains usable after later metrics, computations,
  or dataset replacement. Retaining copies is the consumer's memory decision.
- Callbacks run synchronously in configured metric order. The report and table
  are populated before notification. Without a callback, reporting does not
  make additional matrix copies.
- Callback or snapshot-copy exceptions propagate and stop the loop. They do
  not become metric failures or erase the completed calculation report. The
  progress bar and warning handler are restored on normal exit or interruption.

Reports start empty, reset at each valid `compute()` call, and clear after a new
dataset is loaded. Only finished attempts appear. `KeyboardInterrupt` and
`SystemExit` interrupt computation without labeling an unfinished metric failed.
An interrupted repeated computation can leave old values for unattempted
metrics in the table; use the current reports to identify finished attempts.

Calculator handles one loaded dataset at a time. The [EEG runner](workflow.md) can
associate window/recording identity with its callback; Calculator does not
invent window indices or implement retry policy.

## Timing and warnings

`elapsed_seconds` uses `time.perf_counter()`, a monotonic clock. It measures the
metric call, diagonal masking, table update, and failure-diagnostic formatting.
It excludes result snapshot copying, callbacks, progress updates, and the final
console summary. Loading and preprocessing happen outside this timing. Shared
intermediates can make later metrics cheaper, so these are timings for the
actual configured execution order, not independent estimator benchmarks.

Each report's `warnings` tuple describes warnings emitted during the attempt:
fully qualified category, message, filename, and line number. Existing filters
and handlers still control visibility and repeat suppression. Ignored warnings
and warnings captured internally by an estimator are not recovered. A warning
promoted to an exception becomes a metric failure, following the existing
exception policy.

These are Python warnings; logging messages, stdout, and stderr are not captured
as structured warnings.

Calculator's own `Caught ...` failure warning is emitted after notification and
is not duplicated in the metric's warning tuple; its details are in `error`.
If the caller promotes this warning to an exception, computation stops with the
failed report and callback already published. Callback warnings are outside the
metric attempt and are not attributed to it.

Warning observation uses Python 3.12's process-global `showwarning` hook without
changing warning filters. Concurrent threaded computations are unsupported;
process-based execution remains a subsequent runner task.

The existing printed results summary classifies matrix values by their NaN
counts. Use `metric_reports` for calculation success/failure; the summary
cannot distinguish a successful all-NaN result from a caught exception.

## Calculation completion and storage

Notification means calculation finished, not that values were saved durably.
A writer failure remains a consumer failure even when its input metric
succeeded. The [serial EEG workflow](workflow.md) supplies bounded batch output,
write completion, explicit retries, and coarse resume orchestration around this
callback. Calculator itself remains independent of EDF and durable storage.

The existing `write_calculator` adapter continues to save table values without
consulting these reports. Its behavior and storage format are unchanged; see
[the storage guide](storage.md#explicit-calculator-adapter).

Focused tests are in `tests/unit/test_calculation_reporting.py`. Shared-cache
golden Calculator passes run with and without callbacks, including the extra
wavelet set. See [the test guide](../tests/README.md) for verification commands.
The [verification receipt](../tests/calculation_reporting_results.json) records
866 default passes (16 skips, nine expected failures), 880 full/slow passes
(ten expected failures), 50 focused installed-wheel passes, and 408 bit-identical
pre-change comparison matrices. These are regression-preservation checks, not
independent mathematical validation of every estimator.
