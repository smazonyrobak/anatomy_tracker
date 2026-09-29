# Canonical rehearsal001: synthetic retention recovered, whole-model gate failed

Completed source `0e6dc0d`, 2000 additional AdamW updates (B8000→C10000),
594.692 s. Independent CPU audit passed integrity and failed the predeclared
scientific/engineering advancement gate. **C10000 is not promoted.** All12 gates
were retained: only real donor-macro normal retention against B8000 failed.

The continuation restored the whole experimental B8000 model, optimizer and RNG;
LR changed to0.00025, with synthetic/real loss weights2/3 and1/3. It retained the
matched/canonical real-chart mixture, all three synthetic observation modes and
the complete98304-cell/two-representation evaluation gallery. This combined
continuation does not isolate replay weighting, new replay draws or LR effects.
No independently trained encoders/refiners were merged.

## Fixed endpoint comparison

Capture means a top-K candidate within10° antipodal normal and500µm signed-aligned
normal offset. It is not exact-cell, full-frame, dense-anatomical or electrode
accuracy. Means below give every donor/synthetic group equal weight.

| Readout | Original A6000 | Bridge B8000 | Rehearsal C10000 |
|---|---:|---:|---:|
| Real64/6-donor normal error, ° | 38.5855 | 9.5406 | 12.4258 |
| Real normal-offset error, µm | 3348.90 | 603.17 | 955.52 |
| Real capture@32 | 13.9394% | 92.1212% | 90.6061% |
| Real capture@128 | 26.3636% | 96.8182% | 96.8182% |
| Eligible synthetic611/40-group normal error, ° | 38.6329 | 38.1746 | 38.2545 |
| Eligible synthetic normal-offset error, µm | 1813.21 | 1891.35 | 1974.71 |
| Eligible synthetic capture@32 | 73.6893% | 72.5189% | 78.1442% |
| Eligible synthetic capture@128 | 89.0880% | 89.1330% | 92.4644% |

Real normal error increased2.885223° versus B8000: the frozen limit is11.540608°
(B+2°), missed by0.885223°. Real capture@32 fell1.515152 percentage points, within
its2-point tolerance; capture@128 was unchanged. Both real improvement gates
against original A6000 still pass. Real MAP finite-frame RMS worsened from
3093.87 to3352.56µm; high coarse plane capture does not solve full pose.

## Every real development donor retained

| Donor (images) | Normal A / B / C, ° | Offset A / B / C, µm | Capture@32 A / B / C |
|---|---|---|---|
| 14452 (11) | 36.1071 / 7.5584 / 7.5060 | 1574.18 / 151.08 / 152.26 | 45.45% / 100% / 100% |
| 15219 (10) | 53.5825 / 10.8236 / 9.5833 | 2794.31 / 278.80 / 294.12 | 0% / 90% / 90% |
| 15336 (11) | 37.3343 / 8.1511 / 8.6248 | 4167.71 / 270.96 / 281.50 | 0% / 90.91% / 90.91% |
| 15439 (11) | 31.5688 / 8.4107 / 10.4861 | 3210.85 / 219.03 / 1345.48 | 0% / 90.91% / 81.82% |
| 15447 (11) | 39.3705 / 12.9767 / 18.9966 | 4844.36 / 2456.70 / 2253.19 | 18.18% / 90.91% / 90.91% |
| 15935 (10) | 33.5494 / 9.3231 / 19.3583 | 3501.99 / 242.46 / 1406.55 | 20% / 90% / 90% |

Versus B, normal error improves for two donors and worsens for four. The largest
increases are15935 (+10.0352°) and15447 (+6.0199°); donor15439 adds1126.45µm
offset error. Exactly one previously captured image is lost (15439), none gained.
All64 images remain in these summaries; no outlier/donor exclusion is justified.

## Synthetic mode tradeoff

| Eligible original mode (rows) | Normal A / B / C, ° | Capture@32 A / B / C |
|---|---|---|
| Brush absent (207) | 39.9272 / 41.0191 / 39.6716 | 66.7917% / 64.5417% / 72.6667% |
| Accurate brush (204) | 38.6720 / 37.3235 / 37.9955 | 79.9583% / 79.4583% / 85.6250% |
| Imperfect brush (200) | 36.7581 / 36.2014 / 37.7873 | 74.6667% / 73.7917% / 77.1667% |

Every populated eligible mode and the overall subset pass their frozen A6000
normal/capture retention gates. B's brush-absence failure is repaired. Relative
to B, capture@32 gains/losses are22/6 absent,17/5 accurate and15/8 imperfect;
54 eligible rows gained and19 lost. However MAP offset rises83.36µm overall,
and imperfect-brush normal error rises1.5859°: better candidate-set capture is
not uniform point-estimate improvement.

The29 censored observations were not deleted: their17-group descriptive
capture@32 changes44.12%→38.24% (B→C), while capture@128 changes61.76%→67.65%.
Across all640 rows, capture@32 changes71.09%→76.41%. Eligible and censored
summaries are separate, not interchangeable denominators.

On all256/58-donor TRAIN queries, normal error changes8.3643°→8.0118° and
capture@32 changes97.3276%→96.8966%. Common-anchor-eligible244 training rows
retain99.3103% capture@32. Canonical near-physical anchor hit@1 changes98.1034%
to97.7586% on all training rows. These are training fit diagnostics, not a basis
for overriding the development retention failure.

## Decision and limits

The continuation improves synthetic candidate coverage while sacrificing real
normal accuracy beyond the declared tolerance. Do not promote C, remove difficult
donors, relax the gate or launch a result-driven LR sweep. This evidence supports
addressing the documented acquisition-frame mismatch in training data, not
claiming that a scalar replay/LR compromise has solved it. Any next experiment
needs a separately declared hypothesis and unchanged honest evaluation scope.

Real references are weak upstream Allen affines from six repeatedly observed
development donors, not untouched final validation or expert electrode truth.
Synthetic groups derive from one atlas and are organizational IDs, not
independent biological subjects. Neither this result nor the passing integrity
audit establishes native refinement, calibrated probabilities, proprietary-GUI
readiness or superiority to DeepSlice.

## Frozen provenance and reconstruction scope

Run: `I:/AnatomyTracker/runs/joint_v6_canonical_rehearsal_001`.

- Completion SHA256: `be86f269a97fb99275710fd80a95e41d9a742925c8b4d16ae6371d67a8f1cdeb`.
- Independent audit SHA256: `8af7c7e208326de15555ec3fc90290d5a8ab2579fd77a80f0dfb8ffebc14934b`.
- C10000 whole checkpoint: `888ad9cc493453ce769628fda1aebe70813c031f7809192076c08b000720d9af`.
- C10000 gallery: `76ade1c8ec87d2b995832760f90e0ea58a7593bb00a42d1dc9a9bd9b2abce8b9`.
- B8000 whole parent: `c6aab521d86eee312f12327965b71c6da1d88f184b7e3427dd7ba2782c88c9f8`.
- A6000 original reference: `280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2`.

The independent endpoint audit verified51 frozen nonretrieval tensors,18 changed
retrieval tensors, exact2000 applied continuation updates and raw-score
reconstruction (maximum log-posterior discrepancy3.497209e-6); gate false.

Additional descriptive reconstruction is preserved at
`I:/AnatomyTracker/runs/joint_v6_canonical_rehearsal_001_analysis/`:
`analyze_rehearsal_001.py` SHA256
`88178f00162d7d88c349fe0584386e5529c772eb7293e66b89f0419585e2a0bc`, and
`comparison.json` SHA256
`6c6bc192c0f7158e85f61509324a41b03e300dbd9219097baa8bf5cf499ebae5`.
It authenticates all six raw-row files, checks identical IDs/reference geometry,
and independently recomputes NumPy normal/offset/capture values from saved
top128 cell IDs and catalogue geometry. Full logits/top128 are covered by the
separate endpoint audit; this analysis does not rerender, re-encode or rerank.
All per-group/per-mode comparisons are retained in the JSON, not only examples
selected for this note. Active native signed-pose outputs were not accessed.
