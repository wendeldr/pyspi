# Already-documented calculation issues (cite, never present as new)

Record these in a unit's `already_documented` field with the reference shown. A unit record may
elaborate on them, but `new_finding_candidates` must not repeat them and `new_finding` is never
true for them.

## tests/README.md "Known bugs (locked, not fixed)"
1. `TransferEntropy(estimator='symbolic')` falls through to the Gaussian calculator (`te_symbolic_* == gc_gaussian_*`). — "tests/README.md bug 1"
2. `ConvergentCrossMapping(embedding_dimension=None)` picks max E (10), not best-rho E. — "bug 2"
3. `SoftDynamicTimeWarping` ignores `global_constraint` (three identical configs). — "bug 3"
4. `SpearmanR`/`KendallTau` mutate the class-level `labels` list. — "bug 4"
5. PLI produces signed, antisymmetric values despite its unsigned/undirected labels (tests/README.md item 5, current wording; the earlier wording of this item was the MRO bypass of `Undirected.multivariate` by the spectral classes, which REVIEW M1 still documents). — "tests/README.md bug 5" / "REVIEW M1"
6. `Calculator` duplicate-identifier check is dead code. — "bug 6"
7. `Data(procnames=...)` never stored. — "bug 7"
8. `Data.convert_to_numpy` for `.csv` passes `','` as dtype. — "bug 8"
9. `CorrelationFrame(calc_frame)` calls nonexistent `getstatlabels()`. — "bug 9"
10. `utils.convert_mdf_to_ddf` raises under pandas>=3. — "bug 10"

## tests/README.md "Flagged for investigation" and "worth knowing"
- Degenerate outputs of the nine extra wavelet classes on single-epoch input (cohmag/plv = 1, ppc/dspli -inf, dwspli 0). — "tests/README.md flagged: wavelet degenerate"
- spectral-connectivity pinned at 1.1.2 (2.x changes input layout and frequency grid). — "tests/README.md dependency constraints"
- `utils.normalise` default `nan_policy='propogate'` always raises; the spectral cache's first insert is keyed by `self.measure` (one redundant recompute); `dtf_* == pdcoh_*`; auto-embedding TE kraskov equals fixed k=1 on cml7. — "tests/README.md worth knowing"

## docs/storage_comparison/REVIEW.md (M1–M9)
- M1 generic Undirected mirror bypassed by spectral/cov/pdist/pec classes.
- M2 precision, Euclidean distance and envelope correlation show last-bit reciprocal asymmetry (pinvh, dot-product distances, row-wise normalisation).
- M3 PLI and wPLI are the SIGNED variants (E[sign Im S], E[Im S]/E[|Im S|]); the library documents this; canonical versions take a magnitude.
- M4 coherence phase is an arithmetic mean of principal angles; DC bin included and Nyquist excluded for even n_fft; apparent antisymmetry offset 2π/n_bins from a negative real DC cross-spectrum.
- M5 PSI (spectral_connectivity 1.1.2) sums every unordered frequency pair in the band (`_inner_combination`), not Nolte 2008's adjacent-frequency formula; sign can differ.
- M6 group delay is directional/antisymmetric (reverse slope = negative); all-NaN cases observed in saved data.
- M7 negative dspli/dswpli/ppc values are valid (bias-corrected estimators).
- M8 pec/pec_log can be negative despite "unsigned"; Gaussian joint/conditional entropy can be negative; ce is H(i,j)−H(i) (conditioning order); spearman/kendall label mutation; tlmi is asymmetric by construction.
- M9 label structure discussion.

## Other documents
- `DEPENDENCY_COMPATIBILITY_REVIEW.md`: band forwarding and Nyquist handling.
- `EEG_OPTIMIZATION_NOTES.md` §review: standard vs historical PSI, band forwarding, wavelet PSI axis selection (pyspi indexes the time axis of the (N,N,1,T) PSI result with frequency indices), signed PLI/wPLI, phase aggregation/endpoints, parameter-specific labels, legitimate negative estimators, automatic CCM embedding selection, SciPy 1.18 / ANM drift, pyEDM 2.x CCM drift.
- `DEPENDENCY_MIGRATION.md`: SciPy 1.18 changes `anm` by 1.25e-6 relative; pyEDM 2.x changes CCM values.
