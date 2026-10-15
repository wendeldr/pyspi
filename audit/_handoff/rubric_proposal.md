# Proposed rubric clarifications (superseded 2026-10-15: see README "Score columns: original, rubric reading, adjusted")

Outcome: the user kept the approved rubric and the original `score_0_10` unchanged and asked for extra rules to live in separate columns with documented reasons. An independent adversarial review (fresh agent) recommended how; its recommendations were followed. R1, R2, R3 and R5 (reworded) are applied only through `score_adjusted`; R4 and R6 are README reading rules/glossary; R7 changed the anchor-5 wording. The literal reading of the anchor table is `score_rubric`. The text below is the original proposal, kept for the record.

Raised by the adversarial verdict challenge (Workflow 4b/4b-2). The approved rubric in audit/README.md is unchanged.

Rubric application rules (settled at synthesis on 2026-10-15, after the adversarial verdict challenge):

- **Representative identifier.** A unit's verdict and score describe its representative identifier (the `representative` column, which labels the CSV row). Identifiers of the same unit that behave differently are named with their own anchor in `what_changed` or `check_cost` (for example the three EEG PSI bands that are all-NaN at T=1024 sit at anchor 0 while `spectral.psi` scores 2).
- **Floating-point tolerance (anchor 10).** An entry agrees when |difference| ≤ 1e-8·|reference| or |difference| ≤ 1e-12; the absolute floor covers entries that are zero up to rounding (cosine distance of near-identical rows).
- **Documentation source.** The supplement is the documented definition; where it is silent the GitBook page is. Anchor 10's "matches the pyspi supplement description" means whichever of the two documents the unit.
- **Documentation wrong, computation canonical.** When the code reproduces the canonical definition but the documentation describes a different computation, the unit scores 9 and the record says the documentation, not the computation, deviates (`misc.gpfit.DotProduct`, `distance.hhg`, `misc.coint.johansen_order-0_*`).
- **Transcriptions.** A reference written from the delegate's own source code counts as same-lineage evidence even when it does not import the delegate. A score above 6 must rest on at least one reference derived from the paper or a textbook, an analytic case, or an external package from another lineage.
- **Verdict words.** `wrong`: the computed formula is not the quantity the code or its delegate cites and is not a coherent alternative definition (for example `spectral.dcoh`'s mixed powers). `related-quantity`: a coherent, definable quantity that differs from the canonical one (all-pairs PSI, the arithmetic mean of wrapped phases). `degenerate`: constant, all-NaN or wrong axis.

- **Anchor 5 wording.** Change "verified only qualitatively" to "checked only qualitatively" (wording only; same meaning), in line with the rule that nothing should read as mathematically verified.
