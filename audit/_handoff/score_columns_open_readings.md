# Score columns: open readings raised during classification (not applied)

Source: `_handoff/score_columns_classify.json` (run wf_b6a183cb-5de, 2026-10-15): one classify agent and one adversarial check agent
per batch, then one cross-batch consistency agent. The agents were told to apply the adopted spec exactly (README "Score columns:
original, rubric reading, adjusted") and to report, not apply, any other reading of the anchor table. This file lists those reports
so that each number's alternative readings stay visible. No score in the records was changed because of them.

Why they were not applied: the user asked that the rubric clarifications follow one independent adversarial review, and that review
fixed which rows change (seven `score_rubric` rows, four `score_adjusted` rows, one `verdict_adjusted` row). Applying further readings
would re-score rows that review did not examine. They are inputs for the later per-metric decisions.

The most common theme: several rows sit at 7 as `nearest-anchor` because anchor 7's "matches the pyspi-supplement variant" holds
only in part (cce gaussian/kernel/kozachenko with n = 5 against the documented 10, te.gaussian_auto's factor 1/2, cov.GraphicalLasso,
cov.OAS, spectral.icoh, johansen order-1, misc.ids, pdist.braycurtis). Read strictly, the highest anchor whose bar every condition
clears could be 6 wherever anchor 6's "deviates from both" holds. The `score_rubric_note` of each such row names the clause that fails.

Afterwards the residual-fix workflow (run wf_10e9433e-b12, score-narrative area) appended a pointer sentence to `what_changed` in
20 records, 12 of them units listed here. The section `score_columns_followups` at the end lists them. The readings themselves
stay unapplied; no score column changed.

80 reports on 61 units (one added at the Workflow 4c close, under distance.bary.dtw).

## base.pair_handling

- (b01_base_preproc_cov) The notes give the score as 'rubric 9: the convention differs from the developer guide', which is GitBook text. Under R3, GitBook does not override the supplement, and the supplement's §3.1 'repeated entries' sentence holds: max|A - A.T| = 0 on 15 mirror classes, and the toy run has max abs 0. The 9 then rests only on the cache visiting-order coupling (coint_aeg up to 7.41, xcorr up to 0.081). The record lists that coupling under this unit's deviation_from_canonical, but it arises inside the xcorr, coint and je caches. If the coupling is charged to those SPI units, R3 would give score_adjusted 10. Not applied; please decide.

## basic.cov.EmpiricalCovariance

- (b01_base_preproc_cov) 'misdescribed' rests on the §1.1.1 sentence 'equivalent to correlation'. The whole gap is the ddof factor (T-1)/T (doc-corr max rel 0.01563 at T=64), and the canonical MLE agrees to 1.22e-15. With 'misdescribed', R4 would read this row as nearest-anchor. The basis is anchor because anchor 9 names ddof (SPEC Basis guidance). If the factor counts as an unstated ddof convention instead, the status becomes 'described' and the basis stays anchor. The score is 9 either way. LedoitWolf, ShrunkCovariance and OAS use the same sentence; their gaps also include the shrinkage (0.803, 0.886 and 0.803 of r at T=64).

## basic.cov.GraphicalLasso

- (b01_base_preproc_cov) The basis was changed to nearest-anchor because anchor 7's 'passing optimality checks' fails on some inputs: KKT 34/37, closed form 16/19 and R glasso 10/11, with the stall at 8.65e-4 on var2_bivariate T4096 and a FloatingPointError on trending T4096. GraphicalLassoCV is held at 6 for the same failure type at a larger size (KKT 32/37, stall 7.06e-3, 17/35 refits not converged). A strict reading might find that anchor 6 holds fully here: the diagonal is unpenalised unlike Friedman 2008, and the supplement says 'sparsify the covariance matrix'. If so, the row would be 6. Not re-scored.

## basic.cov.OAS

- (b01_base_preproc_cov) The basis is kept as anchor 7 ('matches the pyspi-supplement variant'), as the record has it, because sklearn 0.24.1, the version the supplement names, computes the no-2/p formula. But the row is 'misdescribed' because the supplement text cites Chen et al. [5] (Eq. 23 keeps the 2/p terms) and says covariance equals correlation (pyspi/r = 0.8034 at T=64). LedoitWolf is nearest-anchor because of the same correlation sentence. If the version citation counts as metadata (SPEC lists versions as metadata), the supplement's variant is Eq. 23 and anchor 7's first clause fails. The cov_prec_simple note lets 7 stand because the gap is a monotone factor. The score is SPEC-decided (R5-examined, 7); only the basis is in question.

## basic.xcorr.max_sig-False

- (b02_prec_rank_xcorr) documentation_status is set to misdescribed. The supplement's class text (§1.1.5: the 'max' and 'sig-False' modifiers, 'Pearson correlation ... in [-1, 1]', 'first T/4 lags', MLE cross-covariance normalised by the autocovariance) describes this computation. pyspi differs from it by T/(T-1) (1.0159 at T=64) and by the one-lag window shift. However, the identifier xcorr_max_sig-False is not in the supplement's SPI list (items 19-21) or on the GitBook page. It comes only from audit/env/retained_metrics.txt:186,188. If 'silent' is read at the identifier level rather than the modifier level, the status becomes silent. R3 would not change the number, because the GitBook has the same text.
- (b02_prec_rank_xcorr) The basis is now nearest-anchor because the end-lag run breaks anchor 8's monotone clause (0.7596 vs 0.1054 by channel order at T=1024, canonical 0.7589 in both). SPEC.md has a precedent where a constructed probe lowers score_0_10 itself: the pdist.cosine near-row probe moved it from 10 to 9. By that precedent this row would fall below 8. The highest anchor whose conditions all hold would be 6 (constant factor plus a window that changes the value), the same as mean_sig-False, which shares the factor and the window. The record and notes/xcorr.md:170 keep 8 and treat the end lag as a caveat. Not applied.
- (consistency) The note says anchor 8's monotone clause fails on the constructed end-lag pair (0.7596 vs 0.1054 by channel order), yet the row sits at 8. Under the nearest-anchor rule, the highest anchor cleared may be below 8. The record's 8 rests on every dataset pipeline run equalling the canonical maximum x T/(T-1). Not re-scored.

## causal.ccm.E-1

- (b03_causal) Carried over (agree); also applies to causal.ccm.E-10. The supplement does not define the _diff identifiers (ccm_E-1_diff, ccm_E-10_diff, ccm_E-None_diff): it lists six SPIs, and GitBook names diff without defining it. The code uses mean - mean^T, and the agg runs reproduce it with max|diff - (mean - mean^T)| <= 5.6e-17. The records' what_changed folds diff into the unit's anchor-6 reason (E-1, E-10), and E-None scores it at 4. No record or note gives diff its own anchor, so no identifier exception was entered. A reviewer may check whether anchor 5 ('a non-standard quantity with no literature definition computed as the code states') fits ccm_E-1_diff and ccm_E-10_diff better than 6. A 5 would stay inside the variant range.

## causal.cds

- (b03_causal) New. Anchor 6 needs a deviation from canonical as well as from the supplement. The canonical half rests mainly on ffactor 2 (13 levels) vs Fonollosa §2.1's sfactor 3 (max rel 0.39-0.50 from sfactor alone). The record marks that candidate inconclusive, because the original jarfo CDS code was not consulted. The ddof difference (rel 0.08-0.12) rests on the paper's 'sample variance', which the record calls ambiguous. Further implementation details remain (re-standardisation on |x| < 3, dropping x-bins with <= 12 samples, exact 0 when at most one bin qualifies), so the 6 still has support. If jarfo's CDS used ffactor 2, the canonical deviation would rest on these details alone. No change proposed.

## causal.igci

- (b03_causal) Carried over (agree). documentation_status is metadata-conflict: the 'unsigned' keyword on a signed output, the class/GitBook name 'Conditional Independence', and the cdt version. The supplement's 'difference in differential entropies ... nearest-neighbor estimators' leaves the order (sign) and the reference measure open; it does not contradict them. The record accepts 'nearest-neighbor' as shorthand for the 1-spacing estimator. A reader who used a Kozachenko-Leonenko 1-NN estimator would get a different value, so this rests on that reading. Separately, the 9 rests on continuous data only, and no exact manifest run on tied data exists. On a quantised EEG window, A correlates 0.993 (T512) and 0.998 (T1024) with the difference in zero-spacing counts. Dither changes the sign of A on 80% (T512) of pairs. No change proposed.

## causal.reci

- (b03_causal) Carried over (agree). documentation_status is metadata-conflict because the record says 'none in substance', and the only stated disagreement is cdt v0.5.23 vs 0.6.0. The supplement text (§1.3.4: 'MSE of the linear regression of the cubic (plus constant) of x with y') does not state the [0, 1] min-max scaling of x and y. That scaling changes the number: pyspi's values lie in 0.0095-0.130, bounded by 0.25, while an OLS on pyspi's z-scored data gives MSE in z units. It also turns x^3 into a full cubic in x. The record lists this as an undocumented detail, not as a different statistic, so metadata-conflict was kept. Under a strict reading of 'a reader implementing from the supplement text would get a different number', this row is misdescribed. Then anchor 7's 'matches the pyspi-supplement variant' would also need a second look.

## distance.bary.dtw

- (b06_dtw_lcss_bary) bary.dtw is metadata-conflict and bary.softdtw/sgddtw are misdescribed, under the same supplement sentence ('minimizes the sum of squared distances'). The split holds only if 'minimizes' is satisfied by a local minimiser. DBA stops at a local minimum (strict local minimum on narrowband pair (0,1); 2-12 iterations, never max_iter), while the other two stop before one. Under a global reading, bary.dtw would also be misdescribed: rel_gap_best is up to 44.5 against 8 starts on delayed_noise. That would undercut anchor 7's 'matches the supplement variant'. The record's adversarial review chose the local reading. I keep it, but the documentation_status of the three units and the 7/6/5 split stand or fall together.

- (Workflow 4c completeness critic, added at the close) `identifier_exceptions` across the barycenter units. distance.bary.euclidean
  now lists `bary-sq_euclidean_mean` and `bary-sq_euclidean_max` at 9, because no document defines the squared variants and anchor 10's
  supplement clause fails for them. distance.bary.dtw, bary.softdtw and bary.sgddtw leave `identifier_exceptions` empty, although their
  bary-sq identifiers are just as undocumented (notes/barycenters.md, 'Documented definition' and Open question 2: 'No document defines
  them'). For bary.dtw, scored 7 on anchor 7, the first clause ('matches the pyspi-supplement variant') arguably fails for
  `bary-sq_dtw_mean` and `bary-sq_dtw_max`, which would then read at anchor 6 and be listed in `identifier_exceptions`; bary.softdtw (6)
  and bary.sgddtw (5) need the same check, where the lower anchors may already fit the squared variants. Not applied: the record owner
  decides. No score column changed.

## distance.bary.euclidean

- (b06_dtw_lcss_bary) identifier_exceptions is left empty because SPEC.md decides it: '(representative already 0; none)'. Read literally, R1 would require listing every other identifier whose own anchor differs. The record (what_changed) and the notes (line 168) give such an identifier: bary_euclidean_max 'would score 10 on its own'. It equals the from-scratch numpy reference with max_abs 0 in 32/32 statistic runs. bary-sq_euclidean_mean ((1+r)(T-1)/(2T), max_rel 1.36e-15) and bary-sq_euclidean_max also match fp-exactly, but the record gives them no anchor. If R1's literal reading is meant, the entry would read 'bary_euclidean_max=10 (matches): ...'. Decide which reading applies.

## distance.bary.sgddtw

- (b06_dtw_lcss_bary) Not re-scored. The record gives 5 under anchor 5's second clause. But the notes say 'the quantity is not the documented Frechet mean': SSG stops after 3-5 epochs, and F is 2-43% above DBA on bivariate_gaussian (0.6-15% for 30-epoch SSG). That is a departure from both the supplement ('minimizes') and canonical (Schultz-Jain), and anchor 6 names exactly that. The same reasoning puts bary.softdtw at 6. The notes also say 'the exact replication would allow a higher score on implementation grounds'. Please check that 5 rather than 6 is consistent; the RNG-stream dependence seems to be the only thing separating the two units.

## distance.bary.softdtw

- (b06_dtw_lcss_bary) deviation_from_documented lists only metadata items (labels, the seven-SPI count, undocumented bary-sq). 'misdescribed' rests on two things. (1) The notes' anchor-6 reasoning: L-BFGS-B stops before a stationary point (gap up to 0.0138, gradient still 2-27% of its value at init, bary_softdtw_max moves by up to 0.051), against the supplement's 'minimizes'. (2) The last sentence of what_changed: with gamma = 1, bary_softdtw_max is 'a different quantity from the DTW-barycenter max that the supplement's wording suggests'. The same gap applies to bary.sgddtw. Either deviation_from_documented should name these supplement departures, or the status should be metadata-conflict, which would weaken the case for anchor 6.

## distance.dcorr.biased-True

- (b05_dcorr_hsic_lagged) The basis is set to anchor because SPEC.md decides this row that way (the squared-vs-not clause), and the entry follows that decision. The record suggests nearest-anchor would fit better. Its canonical_definition is dCor_n = sqrt(R_n^2), and its deviation_from_canonical reads 'none'. The stored value matches dcor distance_correlation to <= 9.4e-14 with no convention applied. The record's what_changed and notes/dcorr_hsic.md:263,278 give the 9 only because anchor 10's supplement clause fails: the literal uncentred reading gives 0.814 against the stored 0.2478, and the supplement presents dcorr/dcorr_biased as one statistic on two scales. That is the R4 nearest-anchor pattern, as for gpfit DotProduct, johansen order-0, the precision units and multitaper_csm. hhg is not a valid comparison, because SPEC.md decides hhg as anchor through the variance clause. Under F1, the squared-vs-not clause names the deviation of the sibling dcorr (biased-False, which returns dCor^2). It does not name a deviation in this identifier's own output. The score of 9 is the same either way.

## distance.dtw.constraint-sakoe_chiba

- (b06_dtw_lcss_bary) 'misdescribed' follows what_changed and the notes' anchor-6 reasoning. The r=1 band blocks the shifting the supplement describes: delays 3/5 stay at 0.724/0.731, against 0.098/0.123 under itakura. Two caveats. deviation_from_documented lists only the undocumented width and the spelling. And the supplement says the value comes from tslearn, whose default radius is 1, so a reader using tslearn defaults would get the same number.

## distance.hhg

- (b04_pdist_hhg_mgc_gwtau) Carried over. The anchor basis decided in SPEC rests on an analogy that the record itself flags. The estimator-variance comparison uses the null SD (14225 on a mean of 64514 at n=256, about 22%), but the sampling SD on dependent data was not measured. On digitised EEG windows, pyspi differs from integer-code exact ties by 1.18e-3 to 3.69e-3 relative, and no manifest run on integer codes exists. Not applied.
- (b04_pdist_hhg_mgc_gwtau) New: README text conflicts with the SPEC decision. audit/README.md:81-84 lists hhg among the units that 'fall back to nearest-anchor' because the supplement's operational description differs (the R4 case). SPEC's decided row gives hhg basis anchor, placing the tie-rounding deviation under anchor 9's estimator-variance clause. This batch follows SPEC (anchor). The README list should drop hhg, or the decided row should be revisited, so that the two agree. The record's what_changed and notes/hhg_mgc.md:254,337 also still say 'house mapping', which SPEC 'Other fixes' says to replace.

## distance.mgc

- (b04_pdist_hhg_mgc_gwtau) documentation_status misdescribed is a judgement call (carried over from the classifier, corrected). For misdescribed: the supplement's dcorr section (supplement.txt:224-228) uses 'unbiased estimator' for the double-centred hyppo statistic, so 'Only this unbiased estimator is included here (as in hyppo)' reads as an estimator claim. The global fallback is the column-centred dCorr^{n,n}, which differs from the U-centred unbiased dCor^2 by 1.79e-3 at T=64. Against misdescribed: the sentence follows directly after the smoothing function 'intended to remove any bias introduced by disconnected components', so 'unbiased' may refer to the smoothing. pyspi also does call hyppo.independence.MGC, so 'as in hyppo' holds literally. Under that reading the supplement conflict is only the 'unsigned' keyword, which gives metadata-conflict. The classifier's mention of 'the GitBook LCSS link' is not a supplement conflict: the supplement cites Shen et al. 2020 correctly, and R3 says GitBook never overrides a supplement that mentions the unit. If metadata-conflict is chosen, anchor 6's 'deviates from the supplement' clause weakens toward anchor 7. SPEC records mgc's 6 as anchor 6, which assumes a supplement deviation, so misdescribed is kept. Please check.

## distance.pdist.braycurtis

- (b04_pdist_hhg_mgc_gwtau) Asymmetry with chebyshev and cityblock (carried over from the classifier, refined). Under the table as written, anchor 7's 'matches the pyspi-supplement variant' clause cannot hold for a unit the supplement does not mention. That is the same reasoning that moved chebyshev and cityblock from 10 to 9 in score_0_10. SPEC keeps 7, decided as 'R5-examined unchanged' with no R3 entry, and this batch follows it, recording the row as nearest-anchor with the clause read through GitBook. Also, R3 as written reads only anchor 10's supplement clause against GitBook. So a strict reading would put score_0_10 below 7, and even R3 would not restore it unless R3's text covered the supplement clause of every anchor (7 and 6 too). Not applied.

## distance.pdist.canberra

- (b04_pdist_hhg_mgc_gwtau) New: the record's canonical_definition field is itself the |x|+|y| form with 0/0 := 0 (canonical_ref names Deza & Deza 2016 for the signed form), and pyspi reproduces it to max_rel 1.69e-15 in 15 from-definition runs. If that field is the canonical, anchor 10's numeric clause holds, and R3 would restore 10 exactly as for chebyshev and cityblock. The 9 rests only on the record treating Lance and Williams' non-negative index as canonical and the signed generalisation as a deviation. SPEC keeps canberra at 9 ('numbers rest on value deviations'), and this batch follows it. Either the record's canonical_definition should name the non-negative form, or the 9 should be revisited. Not applied.

## distance.pdist.cityblock

- (b04_pdist_hhg_mgc_gwtau) Carried over. SPEC's decided-row text calls chebyshev and cityblock 'bit-identical to from-definition kernel'. That holds for chebyshev (14 runs, max_abs 0.0). For cityblock the record has from-definition max_rel 3.55e-15 (max_abs 1.64e-11 on values of about 1e3). It is bit-identical only to the same-lineage cdist and to the scipy pdist probe. The R3 reason here cites the record's number. The score is unaffected.

## distance.softdtw

- (b06_dtw_lcss_bary) identifier_exceptions departs from SPEC.md's exact text 'bit-identical to softdtw in 32 runs'. results/dtw_family.json has 16 ident runs for each constraint identifier (kernel 4 datasets x T 64/256/1024, plus pipeline 4 datasets), 32 in total. So the entry says 16/16 for each identifier and keeps 32 as the total. The decided values are applied unchanged: 3 nearest-anchor, adjusted 9, R1, verdict matches.
- (b06_dtw_lcss_bary) documentation_status is metadata-conflict and describes the base identifier softdtw: the class name 'Dynamic time warping' duplicates DTW's. I keep it because R1 makes the row's verdict and score_adjusted describe softdtw, and SPEC withholds 10 for numeric reasons only. Taken alone, the two constraint identifiers would be misdescribed: the supplement says the SPIs 'include the same global constraints as DTW', and masked soft-DTW differs by 2.5-22.6 (itakura) and 57-1303 (sakoe-chiba). Please decide whether documentation_status follows R1 (base identifier) or the whole unit.
- (b06_dtw_lcss_bary) Layout mismatch outside this batch. audit/README.md has no 'Two score columns' section. It now describes three columns: score_rubric, score_rubric_basis/note, and verdict_adjusted next to the original verdict. validate_records.py requires those fields. SPEC.md's layout (score_0_10_basis/note, with the verdict changed in place to matches for softdtw) will not pass that validator as written. softdtw is where the two layouts differ most: under the README layout, verdict stays wrong as provenance and verdict_adjusted is matches.

## infotheory.cce.gaussian

- (b08_it_je_ce_si_cce) Score 7 kept as instructed, but anchor 7 requires matching the supplement variant. The code uses n = 5 where supplement 1.4.7 says 10 (n = 10 gives 11.95 vs 6.12 on var2_bivariate T1024). It also conditions on the present x where the supplement says 'past of x'. The family notes give the reason for the 7 as 'matches Kramer's quantity but with a value-changing parameter that differs from the supplement', which is the reverse of anchor 7's wording. Read literally, anchor 6 (deviates from both the supplement and canonical in an interpretable way) describes the unit. If Kramer's block H(Y^N||X^N) at N = 5 is taken as canonical, as the record's canonical_definition and deviation_from_canonical state, R4 would instead give 9 nearest-anchor (misdescribed). Basis set to nearest-anchor.
- (consistency) Several rows sit at 7 as nearest-anchor because anchor 7 holds only in part: cce gaussian, kernel and kozachenko (n = 5 vs the supplement's 10), te.gaussian_auto (factor 1/2), cov.GraphicalLasso (optimality on well-conditioned inputs only), and now cov.OAS, spectral.icoh, johansen order-1 x2, misc.ids and pdist.braycurtis. Strictly, 'the highest anchor whose bar it clears' could be 6 wherever anchor 6's 'deviates from both' holds. Records were kept; not re-scored.

## infotheory.cce.kernel

- (b08_it_je_ce_si_cce) Same anchor-7 question as cce.gaussian. Because the code uses n = 5 where the documentation says 10, it does not match the supplement variant, so anchor 7's first clause fails as written. Here n = 10 moves the value only by about 0.005 bits (4.379 vs 4.384 on var2_bivariate T1024). A literal reading gives 6; the R4 reading gives 9. Score 7 kept (SPEC R5-examined list), basis nearest-anchor.

## infotheory.cce.kozachenko

- (b08_it_je_ce_si_cce) Same anchor-7 question as cce.gaussian: n = 5 vs the documented 10 (n10 probe 12.97 / 13.95 vs 6.21 / 6.71 on var2_bivariate T1024). A literal reading gives 6; the R4 reading gives 9. Score 7 kept, basis nearest-anchor.
- (b08_it_je_ce_si_cce) No field change. The 7 describes T <= 2004 only. From T = 2005 every term has more than 2000 rows, and the JIDT naive nearest-neighbour routine moves the value 1.97 nat away from canonical KL (cce|var2_bivariate|4096|pipeline|kl) and 2.61 nat (cce|iid_gaussian|4096|pipeline|kl). The population run cce|gauss_ar1_pair|4096|pipeline|truth fails at max_z 13.2. The family score policy (notes/it_kozachenko.md) scores this defect once, on infotheory.est.kozachenko (wrong, 1), and carries it here with no deduction. The anchor table has no rule for a defect that appears only above a sample size. The unit has a single identifier, so identifier_exceptions cannot carry the regime.

## infotheory.est.kernel

- (b07_it_est_mi_tlmi) Agreed and kept: the supplement's 'width of 0.5 standard deviations' is implemented as the box half-width (box side 1 SD). It was not counted as misdescribed because 'kernel width' for a box kernel is normally the radius r (Kantz-Schreiber, JIDT's own KERNEL_WIDTH, the KDE bandwidth). The record's deviation_from_documented and notes/it_kernel.md do list it as a documented deviation. Under a full-side reading, documentation_status would be misdescribed and basis nearest-anchor (R4 case) at the same score 9. The same applies to the 0.25 width of mi.kernel and tlmi.kernel.

## infotheory.est.kozachenko

- (b07_it_est_mi_tlmi) Agreed and kept: documentation_status is misdescribed although the record's deviation_from_documented says 'none in substance'. The supplement only says 'a nearest-neighbour approach'. Above 2000 rows the output is not a nearest-neighbour estimate: a reader implementing KL from that sentence gets 14.19 nats where pyspi gives 9.82 (iid d=10, T=4096). That excludes described and metadata-conflict. This matches how te.symbolic (wrong 1) reads: its record lists a complete documented-vs-implemented mismatch. If the project intends documentation_status to judge the supplement's wording only, and not delegate bugs, this row would become metadata-conflict (JIDT version; 'linear' label vs documented 'nonlinear').
- (b07_it_est_mi_tlmi) Agreed and kept, with the wording corrected: the score of 1 describes only the >2000-row regime. The record says the <= 2000-row computation would rate 9 (jitter convention). R1 covers identifiers, and this infrastructure unit has none, so nothing changes. notes/it_kozachenko.md says the default 0.5 s EEG window exceeds 2000 rows only above 4 kHz sampling, and no *_kozachenko identifier is in the retained EEG set. The classifier said the window 'never goes above 2000 rows', which overstates this.
- (b07_it_est_mi_tlmi) New: anchor 1's text, 'wrong estimator substituted', fits loosely. JIDT keeps the KL formula but feeds it wrong nearest-neighbour distances; it does not swap in a different estimator, as te_symbolic -> Gaussian does. Basis stays 'anchor' because notes/it_kozachenko.md explicitly assigns rubric anchor 1 and rejects anchor 3. A reviewer may still want to confirm this reading.
- (consistency) The record's deviation_from_documented says 'None in substance' (version text only), but the row is misdescribed. That fits the reader test: above 2000 rows the output is not the KL estimate the supplement names (iid d=10, T=4096: 9.82 vs 14.19 nats), as with te.symbolic. The record text should name this, or the status should be argued again.

## infotheory.est.ksg1

- (b07_it_est_mi_tlmi) New, not applied: est.ksg1 sits at 9 partly because 'on tied data the seeded jitter decides the value'. mi.kraskov and tlmi.kraskov carry the same jitter on the same calculator and are at 10. They reach 10 because they also agree on tie-free data without jitter (mi_ksg1 no-jitter 3.6e-15, sklearn 0.0-1.3e-15). So the jitter alone does not hold est.ksg1 at 9. The reason that remains is psi(N) under a Theiler window, which only affects the DCE units; notes/it_kraskov.md item 6 says this convention holds est.ksg1 at 9. No number changes. The note is only so readers do not take the jitter alone as a reason for 9 elsewhere.

## infotheory.je.kernel

- (b08_it_je_ce_si_cce) documentation_status for je.kernel, ce.kernel and si.kernel is set to metadata-conflict. The conflicts are the JIDT version (v1.5 vs 1.6.1), the 'undirected' keyword for ce and the 'k-1' naming for si. The supplement's 'width of 0.5 standard deviations' is the box half-width, which matches JIDT's kernel width r and the radius r of the Kantz-Schreiber box kernel. If 'width' is read as the full width, a reader would compute the width-0.25 value (je bivariate_gaussian T256: 3.7487 vs 3.9823), and the status would be misdescribed. The 9 would not change, because it rests on the ddof and log-base clauses.

## infotheory.je.kozachenko

- (b08_it_je_ce_si_cce) Basis set to nearest-anchor because the notes rest the 9 on 'the jitter convention', and no anchor-9 clause names jitter (the README's convention list under the anchor table does name 1e-8 jitter, but SPEC Basis guidance requires an anchor-9 clause). si.kozachenko is anchor because its notes use the statistical-tolerance clause, although it carries the same jitter shift at T <= 1024 (2.7e-4 on si|logistic_maps_bidir|1024|pipeline|kl). If the jitter shift were read under that clause instead, je and ce kozachenko would be anchor too. For je the shift is 1.3e-7 and population runs reach max_z 1.82; for ce the shift is 1.4e-4 nat and population runs reach max_z 1.27. There is no score question. mi.kraskov and tlmi.kraskov sit at 10 because their KSG values equal no-jitter canonical to fp on continuous data (mi.kraskov kernel with no jitter 3.6e-15). Here the jitter moves the KL value by more than 1e-8 relative on continuous data (je|anm|1024|pipeline|kl 5.5e-8 relative), so anchor 10 fails against canonical KL. The same applies to infotheory.ce.kozachenko.

## infotheory.mi.kernel

- (b07_it_est_mi_tlmi) Agreed and kept: documentation_status metadata-conflict rests only on the §1.4 sentence 'We primarily use v1.5 of JIDT', while the bundled jar is a post-1.6.1 build. That sentence covers every §1.4 unit. The same reasoning applies to tlmi.kernel, mi.gaussian, mi.kraskov and the est.* rows. If other batches marked rows with only this version conflict as 'described', the infotheory rows should be made consistent.

## infotheory.phi.Geo_norm-0

- (b10_it_xme_phi) Read literally, anchor 9 admits 'matches a MATLAB/author reference for optimiser-based units'. Octave running PhiToolbox phi_G files that are identical to upstream (diff -w) matches pyspi Geo to 3.4e-10 (16/16; Geo_norm-1 10/10). Geo stays at 6 only through the README same-lineage cap (README.md:40-41) and R5, since no other-lineage reference agrees on the canonical minimum (shared_envelope T64 is 1.107 nats over). The nearest-anchor notes for both Geo rows depend on that reading. The README could say that the author-reference clause needs a reference other than the delegate's own source.

## infotheory.phi.star_norm-0

- (b10_it_xme_phi) Wording only; the score is unaffected. The record's what_changed (records/infotheory.phi.star_norm-0.yaml:314-315) and audit/notes/phi.md:411 call 9.55e-9 'the largest relative gap'. 9.55e-9 is the absolute gap of the iid_gaussian T1024 pair, whose relative gap is 6.4e-4 (notes/phi.md:130 and :436). The '3.2e-5 of the null SD' comparison uses that absolute value.
- (b10_it_xme_phi) metadata-conflict for star_norm-0/-1 also assumes the optimiser swap is numerically negligible. pyspi and the vendored copy use finite-difference L-BFGS-B / fminunc, while the PhiToolbox the supplement cites uses minFunc with the analytic gradient. Octave fminunc with GradObj on is at most 4.5e-8 from the optimum, the same as the finite-difference path (notes/phi.md:441-446). Upstream minFunc itself was not run; the records say 'size of the effect relative to minFunc not measured'. Any difference would be at the 1e-8 level and would come from a stopping rule (a free choice), so misdescribed seems unlikely.

## infotheory.phi.star_norm-1

- (b10_it_xme_phi) documentation_status for star_norm-1 and Geo_norm-1 is set to metadata-conflict. The only clear conflicts are the keywords 'nonlinear' and 'unsigned' (the norm-1 value goes negative when the divisor is negative, e.g. scaled_offset raw divisor -0.196). For 'normalized (divided) by entropy' the supplement defers to PhiToolbox and does not say which entropy. The code uses (N_c - 1) min_k past-window Gaussian entropy, which can be negative. If a reviewer reads the unspecified entropy as misdescribed, star_norm-1's anchor-7 clause 'matches the pyspi-supplement variant' needs another look.

## infotheory.te.gaussian_auto-MAX_CORR_AIS_kmax-10_taumax-2

- (b09_it_te_di) The basis is nearest-anchor. Anchor 7's 'matches the pyspi-supplement variant' holds for the MAX_CORR_AIS search but not for the statistic: the supplement names the log-ratio F, and pyspi returns F/2. That same factor is why the fixed-lag unit is misdescribed. Read literally, anchor 6 ('deviates from both the supplement and canonical') could also fit. The score stays at 7, following audit/notes/it_gaussian.md (the factor 1/2 is treated as a convention that does not lower the score further) and the SPEC R5-examined list. The reviewer should confirm whether the factor-1/2 convention may satisfy anchor 7's supplement clause.

## infotheory.te.gaussian_k-1_kt-1_l-1_lt-1

- (b09_it_te_di) Classified as nearest-anchor under R4 and F1: the supplement's 'log-ratio of residual variance' gives 2x the pyspi value, and no anchor-9 clause names a factor of 1/2. The record's what_changed and audit/notes/it_gaussian.md:314 still say 'anchor 9: trivial documented convention ... of the log-base kind'. That wording should be updated to match the nearest-anchor basis.

## infotheory.te.kernel_k-1_kt-1_l-1_lt-1

- (b09_it_te_di) documentation_status is metadata-conflict, because the only explicit disagreement is the JIDT version (supplement v1.5, shipped jar post-1.6.1). Two points could turn it into misdescribed. (1) The supplement's 'kernel width' is JIDT's box half-width (recorded as a convention in infotheory.est.kernel and infotheory.mi.kernel). If 'width 0.25 SD' is read as the full box side, a reader would get a different number. (2) The supplement does not say how widths are set per dimension. A reader who gives the next-value box its own SD would differ by up to 0.0049 bits (ref_own_next_width_variant_maxabs_diff). I read (2) as underspecification, not misdescription. The score is 9 either way. This status should be aligned with infotheory.est.kernel and the other kernel rows ({je,ce,mi,tlmi,si}.kernel).

## infotheory.te.kraskov_auto-MAX_CORR_AIS_kmax-10_taumax-4

- (b09_it_te_di) The status is metadata-conflict and the basis is anchor 7. Both rest on reading the tau-max-2 prose (supplement.txt:489) against the tau-max-4 identifier list (supplement.txt:495-496) as a slip inside the documentation. audit/notes/it_kraskov.md:350 supports that reading: tau-max-2 is the gc_gaussian setting. If the prose is taken as the operational description, a reader using tau <= 2 would get different embeddings (gauss_ar1_pair T=256 selects lt=3 and kt=3), making the row misdescribed. Separately, the record's deviation_from_documented (2) says the text implies AIS selection for the target only. supplement.txt:487-491 instead says 'the embedding parameters' and cites the MAX_CORR_AIS method of [32], and the notes treat source and target selection as the supplement's variant. If the record's reading were adopted, this row would also become misdescribed. infotheory.te.kraskov_DCE_auto has the same slip but is misdescribed anyway because of the window rule.

## misc.coint.aeg

- (b11_misc) The notes base anchor 6's 'deviates from the supplement' clause on the 'undirected' keyword (and 'unsigned'). Under the documentation_status rules, and the tlmi precedent, those keywords are metadata, so this row is metadata-conflict. The supplement's operational text ('a t-statistic of the unit-root test on the residuals') does not fix the regressand. The value issue is the hidden orientation by channel index (up to 1.53 on cointegrated T=256 and 60.87 on trending T=4096). Score kept at 6 per SPEC.
- (consistency) The anchor table has no clear reading for a statistic documented as undirected whose canonical form is asymmetric. Canonical EG at a fixed orientation is reproduced to 1.6e-13. The orientation is undocumented and moves entries by up to 60.87. The record's 6 is 'as the record states it'. Kept; basis changed to nearest-anchor.

## misc.coint.johansen_order-1_ardiff-1

- (b11_misc) The supplement misdescription that holds order-0 at nearest-anchor 9 also applies here and to order-1_ardiff-10: 'maximum eigenvalue'/'trace' for the LR statistics, and 'autoregressive lags' for lagged differences. Basis was left as anchor 7, which reads 'matches the supplement variant' as referring only to the 'constant and linear trend' deterministic term (as notes/coint.md does). A strict reading would make both order-1 rows nearest-anchor.

## misc.ids

- (b11_misc) Anchor 7 requires matching the documented variant. Under R3 the GitBook is the documented definition, and it does not state pyspi's z-score input; the record says 'neither the GitBook nor the code says which normalisation pyspi uses'. The 7 rests on treating the paper's reported min-max [0, 8] setting as canonical, but the record also notes that the paper allows other normalisations. pyspi reproduces Eq. 6/11/12 to 4.4e-16 on its own input. If normalisation is an input choice and not part of the canonical definition, R3 could support a higher number. Not applied.

## misc.lmfit.BayesianRidge

- (b11_misc) Basis anchor rests on reading the stopping-rule gap as 'within statistical tolerance consistent with estimator variance'. The gap is up to 8.5e-4 at T=64 from the evidence-maximiser MSE, and 5 of 20 evidence runs fail the a-priori 1e-4 bound. That gap comes from a deterministic optimiser tolerance, not from sampling error. If the variance clause covers only statistical comparisons, the row is nearest-anchor 9 (anchor 10's numeric half fails at 8.5e-4). It was kept as anchor because notes/lmfit.md:247 invokes that clause.

## misc.lmfit.ElasticNet

- (b11_misc) The basis and the documentation_status use two readings of the supplement that cannot both hold. The status metadata-conflict assumes that 'the Elastic-Net model ... (with the default parameters) from scikit-learn' points the reader to sklearn's defaults, which the record says the code matches (deviation_from_documented (a); exit-model reference 96/96 at 2.8e-16). Under that reading, anchor 6's 'deviates from ... the supplement' clause fails, and anchor 7 may fit: the code matches the supplement variant but differs from the exact minimiser by up to 3.5e-3 in the tolerance band. BayesianRidge's stopping rule was read the same way. Under the other reading ('Elastic-Net model' means the exact minimiser), anchor 6 holds, but a reader would then get a different number in the band, so the status would be misdescribed. notes/lmfit.md:228 states anchor 6 'although sklearn's documented defaults are used'. Basis, status and score were not changed.
- (consistency) Anchor 7's first clause may hold. The supplement's variant is sklearn's default ElasticNet. A numpy reference with only the stopping rule copied from sklearn agrees on 96/96 runs (max_abs 2.8e-16), and copying a stopping rule is a free choice under R5, not same-lineage. It differs from the exact minimiser by up to 3.5e-3. Read as written, the table could give 7, not 6. Not applied.

## misc.lmfit.SGDRegressor

- (b11_misc) notes/lmfit.md:269 says 'Anchor 6 would be the next candidate' but never says which anchor-6 clause fails. The stopped SGD iterate deviates from the documented OLS fit (excess up to 2.93e-3) and from canonical in an interpretable way, which is close to anchor 6's text. The 5 rests on 'checked only qualitatively', yet quantitative bound checks were run (OLS tolerance 24/48). Score not changed.

## misc.pec.orth-True_log-False

- (b11_misc) pec_orth_abs returns (|c_ij|+|c_ji|)/2, which exceeds |pec_orth| by up to 0.0149. The supplement's 'the absolute of the correlation coefficient was used' could be read as abs of the reported value. Under that reading the abs identifier is misdescribed. documentation_status was set from the base identifier pec_orth as metadata-conflict, because the record says neither source fixes whether abs is applied before or after averaging. The abs identifier was not listed in identifier_exceptions, because it still sits at anchor 7 (the supplement's abs variant, broadband).

## misc.pec.orth-True_log-True

- (b11_misc) The same abs-ordering question applies to pec_orth_log_abs, which exceeds |pec_orth_log| by up to 0.0271. Status was set from the base identifier pec_orth_log as metadata-conflict. No identifier exception was listed.

## preproc.two_array_api

- (b01_base_preproc_cov) Classified 'misdescribed' against the supplement's §1 contract ('Prior to evaluating each statistic, we standardize (z-score) the MTS'), which the record treats as the documented definition. The supplement never names the direct-array API, so 'silent' is a possible reading. The score is unaffected either way.

## preproc.zscore

- (b01_base_preproc_cov) Classified 'described': the supplement says only 'z-score along the time axis'. The §1.1.1 'equivalent' gap is charged to the cov rows; please confirm that split. Separately, the zero-variance guard is a floating-point test. The record says some constant rows pass it, and the preprocessing_base note records a constant 1e-3 channel accepted without detrend at T >= 256 and stored as a constant -0.998 row. preproc.detrend counts the same kind of degenerate-input acceptance against anchor 10, but zscore's anchor-9 basis names only ddof. If the guard is charged to this unit, the basis would be nearest-anchor.

## spectral.cohmag

- (b12_spectral_undirected) New: the anchor-9 squared-vs-not basis may not survive R4. The record's canonical_definition gives Carter's MSC |S_xy|^2/(S_xx S_yy), which pyspi returns (7.6e-13). The supplement sentence ('equivalent of the squared time-domain cross correlation') also describes the squared form. The squared/unsquared disagreement is with the name 'coherence magnitude' and the [59] terminology, a label/citation conflict under R4. On that reading the only value-level mismatch is the grid. That gives 9 nearest-anchor, as for multitaper_csm, if the grid counts, or 10 if the grid belongs to band_selection. Left at 9 anchor because the Basis guidance accepts an anchor-9 clause that names the deviation, and the record names squared vs not.

## spectral.dcoh

- (consistency) Anchor 2 is reached by reading 'sign' as direction (values are non-negative), which is a reading of the anchor, not its literal text. By the F1 guidance this could be nearest-anchor 2 with the same note. Left as anchor.

## spectral.dspli

- (b12_spectral_undirected) Carried over and refined: shared-grid convention. Every per-SPI multitaper unit is marked misdescribed because the supplement's '125 uniformly sampled bins on [f0, fs/2], f0 = 4/T' is not the grid used (512 bins from DC at T=1024; DC alone moves cohmag band means by up to 0.092). For cohmag, phase, pli, plv, ppc, dswpli and wpli, the grid is the only operational mismatch the records list; the rest are labels, the [59] citation, or 'wording'. If the grid were owned only by band_selection and multitaper_csm, these rows would be metadata-conflict. icoh ('imaginary part' vs |Im C|) stays misdescribed either way, and dspli probably does too (grouped with imaginary-coherence weighting it does not use). The anchor-7 basis of dspli, dswpli, plv and ppc reads 'matches the pyspi-supplement variant' with the grid scored under band_selection, as the records do. If the grid counted against each unit, that clause would fail for these four rows. One convention should be applied across b12 and b13.
- (b12_spectral_undirected) New: the closed form departs from the canonical pairwise estimator wherever sign(Im S) = 0. At DC this gives an offset of -0.25/n_bins in fmin=0 bands (-0.00049 for the full band at T=1024). For an identical-channel pair it gives -0.25 at every bin, so -0.25 in every band, instead of 0 (shared_envelope (0,2)). The supplement does not specify the closed form, so this is arguably a library deviation from both the supplement and canonical (anchor-6 wording). The notes keep 7 because the effect on distinct channels is at most 0.25/n_bins. Not applied.

## spectral.dtf

- (b13_spectral_directed) Kept as anchor 7. Read strictly, though, anchor 7's first clause, 'matches the pyspi-supplement variant', fails on the frequency grid. The supplement specifies 125 bins from f0 = 4/T with closed bands. The code uses k/n_fft with open bands, and the inclusive rule differs by up to 5.1e-3 (rep extras). The decided spectral.multitaper_csm row treats this grid as misdescribed and moves 10 to 9 nearest-anchor. The case for keeping anchor: the named examples of anchor 7 (bivariate normalisation, 5 tapers, LWR) exist only in these spectral units, which all use this grid, and its 'LWR instead of OLS' example contradicts the supplement's 'least squares'. So anchor 7 already treats 'supplement variant' loosely, and grid-level departures do not fall outside it. The same reading applies to spectral.sgc.nonparametric, spectral.wilson_factorisation, spectral.sgc.parametric_order-1 and spectral.sgc.parametric_order-20 at 7 (the last two have a k/1026 grid labelled k/1024 and a 125-bin documented grid). Please keep these consistent with the b12 spectral rows.
- (consistency) The record lists a transposed layout (row = target) against 'rows = sources'. That convention comes from the GitBook glossary, and the supplement section does not fix the matrix layout, so under R3 it does not block anchor 7. Basis left as anchor. The same applies to the other bivariate directed spectral units.

## spectral.gpdcoh

- (b13_spectral_directed) Not applied. The pdcoh record moved from 4 to 3 because anchor 3 names 'a conditioning measure computed on a 2-channel model' and anchor 4 does not. The same holds for gpdcoh: var2_chain gives pyspi 0.2031 against a conditioned gPDC of 0.0, so the 'partial' in its label also does not describe the output. notes/spectral_directed.md keeps gpdcoh at 4 because its 2x2 value equals the canonical DC^2 and keeps the population direction, and the anchor table does not state that reason. One distinction the table does support: pdcoh duplicates another SPI exactly (it equals dtf, ident 7/7, 4.4e-16), which is the R6 situation of an identifier that duplicates another (anchor 3), while gpdcoh is a distinct quantity. Anchors 3 and 4 both describe gpdcoh, so please check the pdcoh/gpdcoh pair for consistency.

## spectral.icoh

- (b12_spectral_undirected) Carried over (agree). The 7 rests on the record reading the 'unsigned' keyword as the supplement variant. The supplement's operational text says 'the imaginary part of the complex-valued coherency', which is signed, and R4 treats keywords as metadata. On that reading the code deviates from both the supplement text and Nolte's signed ImCoh, which is what anchor 6 describes. The row's own documentation_status (misdescribed for 'imaginary part' vs |Im C|) already takes the text, not the keyword, as the supplement's description. Score left at 7 per instructions.
- (consistency) Under R4, keywords neither block nor set up a match. Without the 'unsigned' keyword, anchor 7's supplement clause fails, and anchor 6 (deviates from both the supplement prose and Nolte's signed ImCoh in an interpretable way) would be the anchor reading. The row keeps the record's 7. Not applied.

## spectral.pli

- (b12_spectral_undirected) Carried over (agree). The SPEC condition for an anchor-0 identifier exception ('_max band identifier constant 1.0 at the audited T') is not met, so none is listed. The record and notes say the max identifiers are only 'close to constant'. At T=1024 every off-diagonal entry is 1.0 on null, var2, bivariate_gaussian and narrowband data, but strongly lagged delayed_noise pairs stay at 0.2-0.6 and identical channels at 0.0. The null fraction equal to 1 is 0.50-0.53 at T=64 and 0.96 at T=256. wpli, dspli and dswpli max show the same null saturation (1.0 in all null entries at T>=1024), but no coupled-data probe was run for them. phase max saturates near pi (null 3.1413 at T=4096) without being constant. None are listed as exceptions.
- (b12_spectral_undirected) New: pli and icoh are treated asymmetrically under R4. The pli record (deviation_from_documented (3)) says the supplement's 'averaging the sign of the (per taper) phase difference' matches the signed E_k[sign Im S]. Only the 'unsigned' keyword disagrees, which R4 calls metadata. Read that way, pli matches the supplement's operational text and differs from Stam's |E sign| in a value-changing way (anchor-7 wording). icoh is the mirror case: the keyword agrees with the code, the text does not, and it sits at 7. wpli is similar: no abs in the text, and the record calls the weighting difference wording. Scores and bases left as they are because anchor 6 literally names signed PLI/wPLI.
- (consistency) SPEC asks for an identifier exception if a _max band identifier is constant 1.0 at the audited T. The record says the max identifiers saturate at 1.0 on null and coupled data at T >= 1024 but are 'close to constant', not constant: identical channels and strongly lagged pairs stay below 1 (shared_envelope T1024 0.2 vs 1.0), and A[j,i] = -min_f. So no exception was entered. dspli_max and dswpli_max saturate the same way on iid data, but their records do not show them constant on all inputs, so they have no exception either.

## spectral.sgc.parametric_order-1

- (b13_spectral_directed) Anchor 7's example 'LWR instead of OLS' itself departs from the supplement text, which says 'estimated via least squares'. The row is therefore anchor 7 only because the anchor names LWR literally. documentation_status is misdescribed: LWR vs OLS is 0.0030 at T=1024, and the documented 125-bin grid differs from the k/1026 grid labelled k/1024. spectral.sgc.parametric_order-20 is the same (0.0080). No number changes.

## spectral.wilson_factorisation

- (b13_spectral_directed) No number changes. The record's anchor-7 reason ('expectation over 5 tapers') is a property of the input CSM, and spectral.multitaper_csm is held at 9 for that same estimator. The 7 is still supported by this unit's own configuration: no zero padding (residual 2.06e-2 vs 9.1e-5 at n_fft = 2T) and convergence judged on the step size only (logistic_maps residual 0.32/0.38, not converged). documentation_status is misdescribed only through the inherited 125-bin grid. Factorising on the supplement's grid would give a different factor, which is the same reading as the decided multitaper_csm row. Please keep this consistent if the multitaper_csm reasoning is revisited.

## wavelet.icoh

- (b14_wavelet) The row is classified anchor (constant up to round-off) to match the record ('rubric anchor 0, constant: 0 up to round-off') and the notes' 'Constant units' grouping. The values are not bit-constant. The max identifiers reach 1.83e-12 to 2.41e-12 at T=1024 (the 4 failing six runs), which is above R2's 1e-12 absolute floor. Under a strict reading of anchor 0, this row would become nearest-anchor like phase, pli and wspli. The score is unaffected.

## wavelet.morlet_cwt

- (b14_wavelet) Basis anchor depends on the record reading anchor 0's 'constant' as constant along the frequency axis. The transform output still varies with t and with the input, so a stricter reading would make this row nearest-anchor. documentation_status is misdescribed because the supplement's §1.5 preamble ('wavelet transformations to derive spectral matrices', 'Morlet transformation') and its 125-bin [4/T, fs/2] grid give a different number. This matches the decided spectral.multitaper_csm row. However, the record notes that the Morlet sentence in §1.5.4 is stated for psi_wavelet only, that psi uses its own 10-frequency, n_cycles=7 call, and that the nine classes using this transform are absent from the supplement. 'silent' is therefore a defensible alternative.

## wavelet.phase

- (b14_wavelet) The row is nearest-anchor because the record itself says the mean identifiers 'are not literally constant'. Score 0 rests on the record's Workflow 4b judgment that no anchor above 0 fits (p1_morlet_phase.py: the mean phase does not track the fraction of negative lag-0 products). A reviewer could still argue for anchor 1 ('wrong estimator substituted'). The score was not re-scored. The max identifiers are pi, or pi - 1 ulp on anm at T=64, on non-identical pairs, and about 1e-31 on the identical shared_envelope pair; all of these stay at anchor 0.

## wavelet.pli

- (b14_wavelet) The row is nearest-anchor because the mean representative follows the exact-zero pattern of FFT round-off (0.465 to 0.9922), not a literal constant. The value also depends on the FFT implementation: a numpy.fft replay differs by up to 0.027, and re-z-scoring moves it by up to 0.0099. It may therefore vary across platforms, which the notes say is plausible but not demonstrated. Score 0 was not re-scored.

## wavelet.wspli

- (b14_wavelet) Same as wavelet.pli: the unit is bitwise equal to pli and is nearest-anchor for the same reason. Score 0 was not re-scored.

## score_columns_followups

Records whose `what_changed` gained a sentence starting 'Rubric reading is <basis> <score> (score_rubric_note), because ...' in
run wf_10e9433e-b12. The sentence says where the original reasoning for `score_0_10` (kept unchanged above it) names an anchor
clause that `score_rubric_note` says fails. Details and before/after checks: `_handoff/wf3_residual_fixes.json`, area score-narrative.
Units marked * also have an open reading above; the pointer sentence does not settle that reading.

- `basic.prec.EmpiricalCovariance`
- `basic.prec.LedoitWolf`
- `basic.prec.ShrunkCovariance`
- `basic.xcorr.max_sig-False` *
- `distance.dcorr.biased-True` *
- `distance.pdist.braycurtis` *
- `distance.pdist.canberra` *
- `distance.pdist.cosine`
- `distance.softdtw` *
- `infotheory.cce.kernel` *
- `infotheory.cce.kozachenko` *
- `infotheory.est.kozachenko` *
- `infotheory.te.gaussian_k-1_kt-1_l-1_lt-1` *
- `infotheory.te.kernel_k-1_kt-1_l-1_lt-1` *
- `infotheory.tlmi.kraskov`
- `misc.coint.johansen_order-0_ardiff-10`
- `misc.gpfit.DotProduct`
- `misc.ids` *
- `preproc.detrend`
- `spectral.icoh` *
