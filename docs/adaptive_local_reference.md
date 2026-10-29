# Adaptive local reference

This custom reference preserves one output per retained contact, in exactly the
same metadata order as CAR. Select it at workflow level:

```yaml
dataset:
  data_lake_config: ../../../seeg-data-lake/config.toml
reference:
  method: adaptive_local
```

No patient-specific contact list belongs in this workflow config. Each patient's
`patient_info/electrode` supplies the output axis and `electrodegroup` supplies
shaft membership. The contact name must equal its group name followed by a
positive integer. Groups can contain multiple letters, digits and primes:
`A1'3` with group `A1'` is contact 3. Ambiguous duplicate numbers such as `A1`
and `A01` on the same shaft fail preflight. A group/name mismatch also fails;
the runner does not guess another grouping or reorder the saved matrix.

## Fixed rules

Within each shaft, retained contacts are ordered numerically. Differences in
contact numbers count physical spacing intervals, including missing contacts.
All contacts on a shaft are assumed equally spaced. Neither MNI coordinates nor
signal correlations determine weights. Bad/excluded contacts never contribute
signals to any reference.

Apply these rules in order, independently for each output contact:

1. If the shaft has only one or two good contacts, subtract the CAR mean of all
   retained contacts in the recording, including the output contact.
2. Find the nearest retained contact in each available direction along the
   shaft. If reaching either neighbor crosses two or more missing contacts,
   subtract the mean of all retained contacts on this shaft, including the
   output contact. Only affected outputs use this fallback, not the entire shaft.
3. If only one direction has a neighbor, subtract that neighbor with weight 1.
   This applies at the first/last retained contact, including when physical
   endpoint contacts were excluded. A single missing contact can be crossed;
   larger gaps already triggered rule 2.
4. Otherwise use a two-sided Laplacian. For immediate neighbors the weights are
   1/2 and 1/2. Across one missing contact, use spacing fractions: if left and
   right distances are `dL` and `dR`, subtract
   `dR/(dL+dR) * left + dL/(dL+dR) * right`.

Examples:

| Retained shaft contacts | Output | Result |
|---|---|---|
| A1, A2, A3, A4 | A2 | A2 − (A1 + A3)/2 |
| A1, A3, A4 | A3 | A3 − A1/3 − 2·A4/3 |
| A1, A3, A5 | A3 | A3 − (A1 + A5)/2 |
| A1, A3, A4 | A1 | A1 − A3 |
| A1, A4, A5, A6 | A1 or A4 | Contact − mean(A1, A4, A5, A6) |
| A1, A4, A5, A6 | A5 | A5 − (A4 + A6)/2 |
| A1, A2 only on shaft A | Either | Contact − recording-wide CAR mean |

All reference weights sum to one, so a common additive acquisition-reference
signal cancels. Endpoint subtraction is not attenuated for a one-contact gap.
No synthetic measurement is inserted for a missing contact.

## Interpretation and output

The output is a custom mixture of local Laplacian, bipolar and mean-referenced
signals. Identical contact axes allow direct alignment with CAR, but the spatial
meaning of each signal depends on its recorded rule. This is not a uniform
second spatial derivative or a reconstruction of missing measurements. Broad
shared physiological activity can be removed along with common noise. Fallback
patterns may differ between patients and should be available to downstream
analysis.

Preflight lists the resolved reference per recording. The output contract and
preparation metadata save its version, equal-spacing assumption, shaft order,
contact numbers, per-contact rule/reason, spacing intervals and reference
weights. Preparation also saves the full linear coefficients, shared-source
diagnostics and exact proportional output pairs. Reference settings participate
in default output identity and explicit-name resume compatibility. The storage
reference component is `adaptive_local`, separate from `car` and `laplacian`.

With only one retained contact in the **entire recording**, CAR itself is zero;
its output remains unavailable. With only two recording contacts, CAR outputs
are exact negatives, which the proportional-pair diagnostic records. A CAR
fallback cannot eliminate that mathematical limitation.

The reference is resolved once per recording and applied to continuous data
before window calculations. Shaft means and the global CAR mean are shared
among all outputs that need them. It is independent of metric selection and
output precision. Raw per-window matrices are saved as usual.

## Metadata scan

The current built dataset contains 69 patients, 912 shafts and 9,506 retained
contacts. All shafts have at least three good contacts. All contact/group names
resolve, and all contact axes remain available under this policy. Counts:

| Rule | Outputs |
|---|---:|
| Ordinary Laplacian | 7,640 |
| Bipolar endpoint | 1,824 |
| Weighted Laplacian | 34 |
| Shaft average | 8 |
| CAR fallback | 0 |

No exact proportional pairs were found in the reference operators. This is a
metadata/operator check, not a claim that measured EEG pairs cannot be perfectly
correlated. Reproduce it with [the scan script](../tests/reference_review/scan_metadata.py);
the [receipt](../tests/reference_review/metadata_scan.json) records the input and
implementation hashes. No EEG or existing connectivity is loaded or changed.

The endpoint convention and shaft-average reference each have published
precedent in [Li et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6495648/).
The combination and one-gap threshold above are this project's explicit policy,
not a claim of an established optimal estimator.
