# Per-unit audit record (`audit/records/<unit_id>.yaml`)

One YAML file per unit in `audit/units.yaml`. `build_csv.py` joins these records with
`units.yaml` (identity), `checks/out/results.json` (closeness numbers) and
`profiles/summary.json` (timings) into `metrics_audit.csv`. Keep every value a plain
string, number, boolean or list; no YAML anchors. Wording rule: the regression goldens
in `tests/regression/goldens/` are snapshots of current output and are never described
as verified, correct, or a reference.

```yaml
unit_id: basic.xcorr.max_sig-True          # must equal the file stem and a unit in units.yaml
status: traced                              # traced -> checked -> profiled (set by the stage that filled the block)

correctness:
  metric_name_standard: ""                  # what a practitioner would call the quantity
  canonical_ref: ""                         # original paper: Author Year, Journal, equation/section
  canonical_definition: ""                  # one-line formula or precise statement
  documented_definition: ""                 # pyspi supplement section + page and/or GitBook page, with a short quote
  delegate_lib: ""                          # library==version and function actually executed ("pyspi" if native)
  implemented_as: ""                        # what is computed: preprocessing, expectation domain, band rule, conventions
  implementation_trace:                     # ordered file:line pointers from pyspi into the third-party code
    - "pyspi/statistics/basic.py:103-107 ..."
  deviation_from_documented: ""             # "none" or a precise description
  deviation_from_canonical: ""              # "none" or a precise description
  units_and_range: ""                       # nats/bits, signed?, symmetric/antisymmetric?, can be negative/NaN?
  direction_convention: ""                  # meaning of A[i, j] from the code (confirmed empirically in the check stage)
  symmetry_expected: sym                    # sym | antisym | asym | n/a
  already_documented: "no"                  # "no" or the reference: "tests/README.md bug 5", "REVIEW M3", "EEG notes §review"
  new_finding_candidates: []                # hypotheses to confirm or refute with scripts; never asserted here
  proposed_checks:                          # consumed by the check stage
    - datasets: ["delayed_noise"]
      path: kernel                          # kernel | pipeline | pipeline-detrend
      reference: ""                         # analytic formula / from-scratch implementation / package.function / R package
      reference_kind: independent-impl      # analytic | independent-impl | external-pkg | same-lineage | literature-only
      criterion: exact                      # exact | tolerance | statistical | direction | optimality | qualitative
      expected_outcome: ""
  # --- filled by the check stage ---
  check_method: ""                          # analytic | independent-impl | external-pkg | same-lineage | literature-only | unverifiable
  check_reference: ""                       # checks/refs/<family>.py:<function>, R/Octave script, package==version, paper equation
  check_data: ""                            # dataset codes used
  agreement_criterion: ""
  closeness: ""                             # numbers from results.json, summarised (max abs / rel, convergence, direction agreement)
  verdict: ""                               # matches | variant | related-quantity | wrong | degenerate | unverified
  score_0_10: null                          # original score from the check stage (kept for provenance); null for unverified
  score_rubric: null                        # the README anchor table read as written; score_0_10 above stays the original (provenance)
  score_rubric_basis: ""                    # anchor | nearest-anchor | none (none only when unverified)
  score_rubric_note: ""                     # required for nearest-anchor and whenever score_rubric != score_0_10
  documentation_status: ""                  # described | metadata-conflict | misdescribed | silent
  verdict_adjusted: ""                      # verdict under rule R1; equals verdict unless R1 changes the described identifier
  score_adjusted: null                      # score_rubric after rules R1/R2/R3/R5 (README); equals score_rubric when no rule applies
  score_adjustment_rules: ""                # "" or rule IDs from {R1, R2, R3, R5} joined by ";"
  score_adjustment_reason: ""               # "" or "R2: <sentence>; R3: <sentence>", one sentence per applied rule
  identifier_exceptions: ""                 # "" or "identifier=score (verdict): reason" entries joined by " | "
  what_changed: ""                          # what differs from the standard, when verdict is not matches
  new_finding: false                        # true only for confirmed findings not in the already-documented list
  definition_under_review: false
  check_cost: ""                            # cheap | moderate | expensive
  symmetry_observed: ""                     # sym | antisym | asym | NaN-pattern, measured on the check runs (max|A - A.T| quoted)
  direction_observed: ""                    # what A[i, j] empirically means on the direction datasets (delayed_noise, var2, logistic_maps_chaotic, anm)
  candidate_resolutions:                    # one entry per new_finding_candidates item, in the same order
    - candidate: ""                         # short restatement of the candidate
      outcome: confirmed                    # confirmed | refuted | inconclusive
      evidence: ""                          # run_id(s) in checks/out/results.json, probe file, or numbers
  checks_run:                               # one entry per manifest run family this unit used
    - run_ids: []                           # run_id prefixes or exact run_ids in checks/manifest/<family>.json
      reference: ""                         # refs/<family>.py:function, r/<family>.R, octave/<family>.m
      outcome: ""                           # pass | fail | n/a with the headline number

# Check-stage rules: `check_method` names the STRONGEST independent reference actually run (a
# same-lineage reference (README rule R5 defines when a transcription counts as one) caps the score at 6 and `verdict: matches` needs an independent or
# analytic reference); `closeness` quotes numbers that appear in checks/out/results.json; `new_finding`
# is true only for a candidate whose `candidate_resolutions` outcome is confirmed AND that is not in
# briefing/KNOWN_ISSUES.md; `definition_under_review` is true when the family note says the standard
# itself is contested (both forms recorded). Trace-stage fields may be corrected by the check stage
# when a check refutes them; say so in the family note.

# --- filled by the optimisation stage ---
optimisation:
  complexity_P_T: ""                        # e.g. O(P^2 T log T)
  hotspot_1: ""                             # file:function, % cumulative (from profiles/<unit>.pstats)
  hotspot_2: ""                             # or 'n/a: <reason>' when no profile names a second frame
  hotspot_owner: ""                         # pyspi | third-party:<package>
  opaque_share: null                        # builtin-time share of the profiled call; 'n/a: <reason>' when no profile frame exists for the path
  profile_file: ""
  basis_scope_note: ""                      # optional: why this unit's ranking basis only approximates its retained EEG identifiers (untimed band variants, cache order) and
                                            #   the direction of the error where known, or why its eeg_basis holds; build_csv appends it to the basis_scope_note column (README
                                            #   "Optimisation columns", rule 1)
  eeg_basis: family-warm                    # optional, units in the EEG selection: fresh (rank tier 0 on the fresh projection: the family-warm run filled a cache this unit
                                            #   reads on identifiers outside the EEG selection, and an EEG run fills it here) | family-warm (keep the family-warm figure: a
                                            #   retained identifier fills that cache first in an EEG run) | retained-increments (rank on the retained identifiers' family-warm
                                            #   increments although some EEG identifiers were never timed); build_csv flags cross-unit cache cases that name no eeg_basis
  no_opportunity_reason: ""                # required in practice when opportunities is []: why no avenue is listed (refuted, recorded as a cross-cutting row,
                                            #   not material, or the user's decision); build_csv writes it into the opportunity column (start it with 'No opportunity')
  opportunities:
    - opportunity_id: ""
      opportunity: ""
      opp_type: ""                          # cache-sharing | algorithmic | vectorise-batch | replace-library | rewrite-numba-rust-cpp | parallel | gpu
      expected_speedup: ""                  # on today's cost; its lower bound (first number before x) is the ranking k, at or below the ratio the cited
                                            #   evidence gives against the ranking basis (rounded down, never up), in the run order that basis assumes (the
                                            #   EEG order in tier 0, where a later reader of a cache another unit fills gains only on what it still computes)
      speedup_basis: ""                     # profile:<pstats> <frame> <share>% | complexity:<argument> | microbench:<profiles/micro/opp_ file>
      output_risk: ""                       # bit-identical | within-tolerance | semantic | rng-stream | unverified-<class>
      risk_evidence: ""                     # profiles/micro/<file>.json with max_abs_diff etc.
      effort: ""                            # S | M | L | XL
      depends_on: []                        # avenues this one builds on: ranked on the residual they leave on this unit; a throughput id here records a
                                            #   deployment precondition or pairing, not a ranking prerequisite
      subsumed_by: []                       # optional: avenues that contain this one on this unit (it saves nothing on top of them; build_csv credits no saving
                                            #   where such an avenue is stacked in the same plan; one with eeg_applicable: false is not, so in tier 0 this one keeps its k)
      independent_of: []                    # optional: avenues of this unit that cut a different part of its cost (a different factor, or disjoint frames), so this
                                            #   one keeps its own k on the residual they leave
      residual_speedup: ""                  # optional: speedup on the residual after the depends_on avenues, stated in speedup_basis: where they change what this one
                                            #   acts on (1x when nothing is left), or where it is worked out from this unit's own profile while expected_speedup carries
                                            #   another unit's figure; its lower bound replaces expected_speedup's on units where a depends_on avenue is stacked before it
      speedup_kind: per-window              # per-window (default) | throughput (more cores, no per-window CPU saving) | projection (hardware or
                                            #   software not installed here, no code run)
      eeg_applicable: true                  # optional; false when the avenue gives no gain on the unit's retained EEG identifiers (its row leaves tier 0)
```

Optimisation-stage keys are read by `build_csv.py` as README "Optimisation columns" rules 1-5 describe. The link keys
(`depends_on`, `subsumed_by`, `independent_of`, `residual_speedup`) belong to the record, so one opportunity id can carry
different links on different units. A `speedup_basis` or `residual_speedup` that has no micro-benchmark of its own stays
`profile:` / `complexity:` based. Cross-cutting rows in `profiles/crosscutting.yaml` take the same opportunity keys plus
`scope_speedup`, a list of `{scope: [unit-id globs], expected_speedup, reason}` whose first matching entry gives the figure on
those units; a row with an explicit `projected_patient_s` states its own `basis_scope` and `basis_scope_note`.
