import json, sys, collections
r = json.load(open(sys.argv[1]))
by = collections.OrderedDict()
for c in sorted(r['concerns'], key=lambda c: c['unit_id']):
    by.setdefault(c['unit_id'], []).append(c)
L = ["# Score columns: open readings raised during classification (not applied)", "",
"Source: `_handoff/score_columns_classify.json` (run wf_b6a183cb-5de, 2026-10-15): one classify agent and one adversarial check agent",
"per batch, then one cross-batch consistency agent. The agents were told to apply the adopted spec exactly (README \"Score columns:",
"original, rubric reading, adjusted\") and to report, not apply, any other reading of the anchor table. This file lists those reports",
"so that each number's alternative readings stay visible. No score in the records was changed because of them.", "",
"Why they were not applied: the user asked that the rubric clarifications follow one independent adversarial review, and that review",
"fixed which rows change (seven `score_rubric` rows, four `score_adjusted` rows, one `verdict_adjusted` row). Applying further readings",
"would re-score rows that review did not examine. They are inputs for the later per-metric decisions.", "",
"The most common theme: several rows sit at 7 as `nearest-anchor` because anchor 7's \"matches the pyspi-supplement variant\" holds",
"only in part (cce gaussian/kernel/kozachenko with n = 5 against the documented 10, te.gaussian_auto's factor 1/2, cov.GraphicalLasso,",
"cov.OAS, spectral.icoh, johansen order-1, misc.ids, pdist.braycurtis). Read strictly, the highest anchor whose bar every condition",
"clears could be 6 wherever anchor 6's \"deviates from both\" holds. The `score_rubric_note` of each such row names the clause that fails.", "",
"Wording or consistency items from this list that were fixed in the records, notes or README afterwards are marked in",
"`score_columns_followups` at the end.", "",
f"{len(r['concerns'])} reports on {len(by)} units.", ""]
for uid, cs in by.items():
    L.append(f"## {uid}")
    L.append("")
    for c in cs:
        L.append(f"- ({c['batch']}) {c['concern'].strip()}")
    L.append("")
open(sys.argv[2], 'w').write('\n'.join(L))
print(len(L))
