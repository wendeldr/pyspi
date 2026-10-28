# Group-delay missingness and significance review

The original outputs cannot establish whether group delay is useful or
scientifically unavailable: their NaNs do not record a reason, and their
selection procedure had confirmed defects. Following the user's approval,
production significance now uses the zero-coherence null and per-pair BH.
Window length/smoothing remain
0.5 seconds, non-overlapping, NW=3/five tapers by the user's decision.

## Historical output inspection

Read every saved window for five recordings (001, 009, 041, 096, 111), all ten
group-delay bands and all channel pairs. These have 71–249 channels and
602–609 windows. This is a selected sample, not a complete cohort audit.
The source is `/run/media/dan/DansFlash/ram_needs_rma/connectivity` and was
opened read-only outside the sandbox because Zarr hangs inside it. Shard-index
checks and numeric decoding found no missing/unreadable chunks in these reads.
The diagonal was excluded; only one direction per pair was counted.

| Stored band label, Hz | NaN percentage range across five recordings |
| --- | ---: |
| 1–4, 4–8, 8–12 | 100% in every recording |
| 12–30 | 93.53–100% |
| 30–70 | 93.63–97.93% |
| 70–150 | 78.39–98.44% |
| 150–250 | 62.77–98.48% |
| 1–70 | 73.26–92.71% |
| 1–250 | 84.93–97.82% |
| 70–250 | 76.35–97.47% |

Band labels come from the stored metadata. The original preprocessing and
resampling have not been recovered, so those labels alone do not prove the
physical frequency coverage in every original recording.

Many mostly-missing connections do have some finite windows. In the stored
1–250 band, 95–100% of pairs have at least one finite window, depending on
recording. In recording 001, the median pair has only 3 finite windows out of
609. A NaN-skipping average would therefore often produce a number, but it
would summarize selected windows with detectable coherence, not the entire
recording equally. A NaN is not a zero delay or evidence of no relationship.

Reproduce with `audit/checks/probes/group_delay_missingness.py --root PATH`.
Counts and per-pair coverage quantiles are in
`audit/briefing/group_delay_missingness.json`. Reading took about 11 seconds;
this is storage inspection time, not calculation time.

## What can produce a NaN

With 1,024 samples at 2,048 Hz, FFT bins are 2 Hz apart. The original open
endpoints leave only one bin in each of 1–4, 4–8 and 8–12 Hz. Its minimum of
three regression points cannot be met, regardless of signal strength. That
is an incompatible band/window configuration, rather than evidence of weak
coupling. This calculation is conditional on the stated sample rate/window;
it does not reconstruct undocumented historical preprocessing.

The current inclusive bands fix endpoint loss. The already-corrected spacing
uses the transform's 12 Hz smoothing bandwidth instead of band width / 50.
Three retained points must span at least 24 Hz, and must lie in one significant
continuous run. With the current settings, the three narrow bands and 12–30 Hz
still cannot qualify. This is an estimator support requirement, not evidence
that a numerical operation failed. There is no proposal to weaken it merely
to obtain more finite results.

For broader bands, the old installed selector computes coherence p-values, applies
BH across all frequencies and pairs, retains the largest significant run, then
spaces its points and requires at least three. Each selection stage can remove
a candidate estimate. Pooling BH makes a fixed pair's result depend on other
channels; earlier independent audit probes demonstrated finite-to-NaN changes
when unrelated channels were added. The saved arrays alone cannot attribute
each historical NaN to a particular stage or rule out run-time failures.

## Per-pair correction cost and decision

The user accepts correction separately across frequencies for each pair/window,
provided runtime does not become excessive. An isolated vectorized prototype
was compared with the installed pooled BH. Each timing used 6,903 pairs
(118 channels), synthetic p-values, one warmup and five alternating timed runs.
Sampled pairs agreed with SciPy's axis-specific BH decisions.

| Frequency bins | Pooled BH median | Per-pair BH median |
| ---: | ---: | ---: |
| 35 | 7.65 ms | 5.78 ms |
| 125 | 46.89 ms | 28.03 ms |
| 513 | 202.47 ms | 126.05 ms |

The correction itself was faster. These timings exclude spectra, p-value
calculation, cluster selection, regression and I/O. More qualifying pairs may
increase regression work, so they are not full group-delay speedups. The
approved per-pair scope does not control FDR across an entire network. This
scope is now implemented, along with the calibration correction below.
`audit/checks/probes/group_delay_significance_probe.py --part benchmark`
preserves the prototype used in the initial inline timing run.

## A separate confirmed significance defect

The installed Fisher approximation is inappropriate for testing coherence
against zero. In 200,000 independent simulated cases, each with five independent
proper complex-Gaussian observations per signal and no true coupling, its
nominal 5% test rejected 15.847%. The theoretical rate for its magnitude
threshold is 15.725%. The standard squared-coherence zero-null test rejected
5.021% in the same simulation. This check is per frequency, before BH or
cluster selection; it does not calibrate the entire EEG pipeline.

For K independent, equally weighted proper complex-Gaussian observations,
the null squared coherence has distribution Beta(1, K−1), giving
`p = (1 - abs(C)**2)**(K - 1)`. Current upstream
[documents this replacement](https://spectral-connectivity.readthedocs.io/en/latest/_autosummary/spectral_connectivity.statistics.coherence_significance_pvalue.html)
and uses it in its
[group-delay selection](https://spectral-connectivity.readthedocs.io/en/latest/_modules/spectral_connectivity/connectivity.html).
The installed dependency still uses the Fisher approximation; pyspi now owns
this selection implementation. DC/even-FFT Nyquist use the corresponding real
Gaussian null, Beta(1/2, (K−1)/2). For independent real Gaussian K-vectors the
squared normalized dot product is one chi-square degree of freedom divided by
the sum of K, giving those beta parameters. No mean is subtracted across
tapers. Odd-length FFTs have no Nyquist bin. Independent Gaussian checks cover
both cases. Tapers are only approximately independent and the complex model
only approximately proper on real spectral data, particularly near DC/Nyquist.
These assumptions and dependence between frequency tests mean this is not a
demonstration of exact 5% FDR in arbitrary EEG.

This defect is too permissive, so correcting it can increase missingness. It
cannot explain the original NaNs as an excessively strict zero-coherence test.
The approved correction distinguishes unsupported bands, insufficient selected
points and numerical failures. Alpha remains 0.05; the first largest continuous
significant run, transform-derived spacing, and three-point minimum remain.
Unwrap only the selected continuous run before spacing its points: this keeps
the same phase-slope definition while avoiding contamination from NaNs outside
the run. A free-intercept slope is unaffected by a constant 2π phase offset.

The receipt is `audit/briefing/group_delay_null_calibration.json`; reproduce
with `audit/checks/probes/group_delay_significance_probe.py --part calibration`.
These are focused diagnostics. Unrelated test suites and the window/smoothing
pilot were not rerun. The calculation-review CSV records only short summaries
of implemented changes and measured timing.

## Implementation, diagnostics and complete timing

`pyspi/_group_delay.py` batches BH, contiguous-run searches and centered OLS
across pairs/windows, with bounded per-block temporaries. It evaluates each
unique pair once and fills the reverse delay/slope/r with the opposite sign.
Delay, slope, rvalue and diagnostic arrays share one bounded cache entry.
Invalid/nonfinite coherence is never significant; material magnitudes above
one are flagged while normalization roundoff at unity is tolerated. Invalid
bins remain unrejected members of the original band's testing family.

`GroupDelay(...).diagnostics(data)` returns detached matrices:

- `reason`: diagonal, estimated, no_significant_frequencies,
  insufficient_frequency_span, invalid_coherency, or numerical_failure.
- `retained_points`: count after spacing the largest significant run.
- `invalid_bins`: count of nonfinite/out-of-range coherencies in the band.

The reason describes the delay fit. A constant phase yields zero delay but an
undefined rvalue. A band too short to support three points raises a descriptive
ValueError, already recorded by Calculator as a failed calculation. Diagnostics
do not change the shared MetricReport schema and are not automatically persisted
by the workflow; they can be collected from the SPI with the same Data object.

Three alternating timing repetitions compared fresh Data/FFT/coherency plus
three bands (0–1024, 0–512, 512–1024 Hz) on 118×1024 input against `cb60808`:

| Fixture | Previous median | Corrected median | Speedup |
| --- | ---: | ---: | ---: |
| One strongly coupled pair amid independent channels | 5.94 s | 1.06 s | 5.6× |
| All pairs strongly coupled | 13.19 s | 1.94 s | 6.8× |

Both versions returned 1/1/1 finite pairs in the sparse case and
6903/6903/6903 in the dense case. The speedup therefore does not depend on fewer
regressions in these fixtures. Results intentionally use different significance
definitions; this is not a numerical-equivalence benchmark or an EEG runtime
prediction. I/O and other metrics are excluded. The receipt preserves the timed
source hashes; a subsequent overflow-only guard converts unrepresentable slopes
to NaN with a numerical_failure reason and has its own focused check.
Reproduce with `audit/checks/probes/group_delay_corrected.py timing`.

## Corrected EEG sample

Evaluated 20 evenly spaced half-second windows and 12 label-blind contacts
(66 pairs) in each of the same five recordings. This uses the preceding pilot's
preprocessing: resample to 2048 Hz, notch/filter/CAR, then window detrending and
normalization. Original preprocessing remains unverified, so this is not an
exact before/after comparison to the stored outputs. No EZ labels were used.

The 1–4, 4–8, 8–12 and 12–30 Hz bands are structurally unsupported with the
retained window/smoothing. Across the other bands:

| Band, Hz | NaN percentage range across five recordings |
| --- | ---: |
| 30–70 | 98.86–99.92% |
| 70–150 | 97.65–99.92% |
| 150–250 | 94.47–100% |
| 1–70 | 97.58–99.77% |
| 1–250 | 95.45–99.24% |
| 70–250 | 96.14–99.92% |

There were no invalid-coherency or numerical-failure outcomes in these sampled
windows. In 1–250 Hz, of 6600 pair/window estimates, 154 had a delay, 4861 had
no significant frequencies, and 1585 lacked a long enough significant run.
These outcomes identify the corrected selector's behavior, not proof that
physiological coupling is absent. With the chosen short windows, missingness
remains a practical reason to omit GD from the user's analysis. Inspection of
the historical `pyspi/workflow/retained185.yaml` confirmed that it included ten GD band
configurations; the earlier statement that it already excluded GD was incorrect.
Preset membership was not changed during this experiment. The subsequent config
reconciliation removes all ten from `pyspi/workflow/reviewed.yaml`; corrected GD
remains available for explicit configurations.

Reproduce with `audit/checks/probes/group_delay_corrected.py eeg` outside the
sandbox for Zarr/MNE access. Receipts are
`audit/briefing/group_delay_corrected_{timing,eeg}.json`. The EEG check evaluated
only group delay, not the prior smoothing experiment or other metrics.

## Validation

37 focused unit checks passed: the new group-delay reference/diagnostic tests
and affected group-delay cases from the spectral, band and direction tests.
Six affected regression cases passed, three each at small and full fixture
sizes. The initial 36 unit cases were run once; the later overflow guard had
one additional targeted test. No unrelated suite was rerun. Six GD snapshots
were refreshed after the independent mathematical checks; all unrelated arrays
were verified byte-for-byte unchanged, and prior snapshot provenance retained.
Only the three GD CSV note fields changed; user flags and other records were
preserved.
