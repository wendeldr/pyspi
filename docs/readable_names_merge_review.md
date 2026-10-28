# Readable names integration review

Integrates `readable-names` at `28726da` with the completed calculation work at
`4f8a56f`. The review covers naming, Hz conversion, structural feasibility and
integration with recent native implementations; it is not a new scientific
review of every estimator. Wavelet calculations are outside the user's intended
run and were not expanded into a separate refactor.

## Result

- Output identifiers use the new parseable spelling, including the newer
  histogram, band PEC, PAC, IDS and constrained-DTW options. Examples are
  `cohmag_mean_band0030-0070hz`, `pec_orth_abs_band0030-0070hz`, and
  `pac_phaseband0016-0030hz_amplitudeband0070-0150hz`.
- The reviewed EEG spectral preset specifies `fs: 2048` and Hz bands. Equivalent
  cycles-per-sample and Hz calculations agree within 1e-12 across every selected
  spectral configuration. No band edges, windows or smoothing were changed.
- The current selections remain 188 main, 12 amplitude, 10 CAR candidates and
  9 Laplacian candidates. All pass workflow preflight at 2048 Hz.
- Invalid window/band combinations stop before sample preparation or output
  creation. The error identifies the offending metric, available frequency grid
  and required bins. It does not silently widen bands, drop metrics or change
  window lengths. Calculator also rechecks before computing, covering metrics
  added after loading data.

## Corrections made during review

Conflicts in distance correlation, PEC/PAC, IDS, tests and goldens were resolved
while retaining the latest native calculations, band inputs, cache behavior and
Java dependency decisions. Candidate classes added after the worktree fork now
use the same naming grammar. Duplicate Calculator identifiers raise an error
instead of overwriting a requested output. Integer and float spellings avoid
precision collisions and normalize negative zero.

The worktree's empirical minimum-window rules for GraphicalLasso and parametric
spectral Granger were not mathematical impossibility thresholds. They were
replaced by structural constraints; regularized estimation is not rejected just
because channels outnumber samples. PEC/Bray–Curtis restrictions that assumed
normalization were not applied to valid unnormalized inputs.

The native corrected distance-correlation minimum is four samples, not the old
wrapper's three. Kernel checks now account for aligned sample counts and fixed
temporal exclusions. Local-reference checks count available contacts, not masked
output slots. Data-dependent singularities, convergence failures and automatic
exclusion choices remain runtime diagnostics: passing preflight is not a promise
that every signal produces finite estimates.

The existing wavelet-PSI wrapper indexes time with frequency indices; its
Calculator/workflow use is explicitly refused rather than certified by an
incorrect frequency-count check. No replacement wavelet estimator was introduced,
and this does not affect the selected multitaper PSI calculations.

## Evidence

- Initial focused naming, feasibility, metric-resolution, preflight and
  Calculator checks passed apart from the subsequently removed PEC restriction.
- 132 affected native calculation, grid, band, naming-integration and early-stop
  checks passed.
- 47 selected regression/catalogue, preprocessing and Hz-equivalence checks
  passed; 20 native-dependency, loader and estimate checks passed.
- Five additional native-kernel alignment/exclusion boundary checks passed.
- Three stored-output checks passed: serial/spawn/resume consistency,
  calibration output, and band metrics with reference masks.
- [Migration receipt](../tests/regression/goldens/readable_name_migration.json):
  734 retained golden arrays preserve their dtype, values and NaN masks exactly.
  The worktree's twelve unsupported single-epoch wavelet fixtures are explicitly
  listed as removed. No retained numerical golden was recalculated.

Historical benchmark receipts and existing output stores retain their recorded
identifiers. [The mapping](metric_names.csv) connects those names to the current
ones, including the candidate selections. New output uses the new names; old
names are not executable compatibility aliases. A renamed selection changes the
workflow contract, so use a new run name rather than resume an old-name store.
