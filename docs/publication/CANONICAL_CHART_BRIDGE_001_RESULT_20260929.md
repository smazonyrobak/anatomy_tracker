# Canonical chart bridge: large paired gain, failed retention gate

Run `joint_v6_canonical_bridge_001` exited successfully after 683s. Its independent
CPU audit passed integrity, but **both arms failed the predeclared whole-model
adaptation gate**. Arm B is a promising coarse-retrieval candidate, not a qualified
replacement, calibrated model, native registration result or benchmark winner.

Both arms continued the **same complete A6000 checkpoint** for 2000 updates using
identical real/synthetic rows, negatives, initial optimizer/RNG state and losses.
A always used the matched continuous affine atlas key. B used exactly8000 matched
and8000 canonical key choices before eligibility weighting. Canonical keys retain
the exact plane and source roll, but use the catalogue's projected centre, 12mm
orthogonal spans and zero shear. Both charts had to have support≥64 for a real
training row to contribute;244/256 rows qualified. All256 remained in evaluation.
No encoder merge, alternate real-image preprocessing or nearest-cell pseudo-label
was used. The shared eligibility and physical-plane negative controls differ from
the older failed real+synthetic run; **B−A here** isolates chart presentation.

## Real development results

Equal-donor means over64 images from six development donors:

| Metric | Frozen A6000 | Matched control A8000 | Bridge B8000 |
|---|---:|---:|---:|
| MAP normal error | 38.585° | 38.420° | **9.541°** |
| MAP normal-offset error | 3348.9µm | 2968.9µm | **603.2µm** |
| Plane capture@32 | 13.94% | 19.85% | **92.12%** |
| Plane capture@128 | 26.36% | 28.94% | **96.82%** |
| MAP finite-frame RMS | — | 9164.1µm | **3093.9µm** |

Capture requires antipodal normal≤10° and sign-aligned normal offset≤500µm.
References are upstream Allen affines, not independently annotated histological
landmarks. The residual finite-frame error is substantial: plane capture is not
successful full-frame alignment, deformation or electrode-site localisation.

| Donor | Images | A normal | B normal | A capture@32 | B capture@32 |
|---|---:|---:|---:|---:|---:|
| 14452 | 11 | 31.20° | 7.56° | 54.55% | 100.00% |
| 15219 | 10 | 40.90° | 10.82° | 0.00% | 90.00% |
| 15336 | 11 | 33.81° | 8.15° | 9.09% | 90.91% |
| 15439 | 11 | 49.63° | 8.41° | 9.09% | 90.91% |
| 15447 | 11 | 36.56° | 12.98° | 36.36% | 90.91% |
| 15935 | 10 | 38.41° | 9.32° | 10.00% | 90.00% |

All six improve in this paired comparison. Donor15447 still has mean offset error
2456.7µm, illustrating why high top-K capture does not certify the MAP result.
These are repeatedly exposed development donors, not untouched final validation.

## Training-gallery closure

Equal-donor all-training means (256 images /58 donors) changed from A to B:

- Full-catalogue normal error36.94→8.36° and capture@32 26.38→97.33%.
- Matched own-anchor hit@1 83.45→69.40%; matched near-physical hit@1 remains98.79%.
- Canonical own-anchor hit@1 1.72→19.83%; canonical near-physical hit@1 37.33→98.10%.

On the244 common-eligible training rows, B capture@32 is99.31%, matched
near-physical hit@1 is100%, and canonical near-physical hit@1 is99.31%.
Here near-physical means normal≤10°/offset≤500µm, **not** the older finite-frame
near-equivalence definition. The paired change supports the acquisition-chart
diagnosis; lower exact matched-anchor discrimination is not hidden. The result
does not establish that canonicalisation solves full pose or uncertainty.

## Exact arbitrary-plane retention failure

Synthetic values are equal **synthetic-group** macros, not independent animal
generalisation. Raw probabilities were independently sorted and geometric capture
recomputed for this mode analysis, without rerunning a model.

| Eligible subset: capture@32 | A6000 | Control A | Bridge B |
|---|---:|---:|---:|
| All | 73.689% | 73.156% | 72.519% |
| Brush absent | 66.792% | 67.875% | **64.542%** |
| Accurate brush | 79.958% | 78.625% | 79.458% |
| Imperfect brush | 74.667% | 73.083% | 73.792% |

B's brush-absent capture drops **2.25 percentage points** versus the frozen parent;
the permitted drop was2.00 points. The required floor is64.7917%, so B fails by
**0.25 percentage points**. Versus matched control A, the drop is3.3333 points.
The narrow miss is not waived or rounded into a pass.

The absent subset has207 eligible rows spanning40 synthetic groups. Parent→B
loses19 captured rows and gains15;14 groups worsen,11 improve and15 are unchanged.
Control A→B loses19 and gains12;13 groups worsen,8 improve and19 are unchanged.
Thus the macro miss is not just one selected row or one group. Group sizes are
small and no confidence-interval significance claim is made.

B's absent normal error39.927→41.019° rises1.092°, within the2° bound. All its
other frozen synthetic retention gates and both real-improvement gates pass.
Control A fails both real-improvement gates and the imperfect-brush normal-error
retention gate. Neither complete checkpoint satisfies the whole-model gate.

A focused next coarse experiment is a predeclared whole-B continuation with
stronger broad arbitrary-plane synthetic rehearsal relative to real adaptation,
explicit no-outline coverage while retaining both brush modes, and a smaller
learning rate. It must retain canonical real positives and unchanged gates.
Do not replay these failed development rows, select specific development groups,
retune on the six donor results, relax the retention floor or splice a separately
trained native refiner into B. This is a proposed experiment, not a promoted model.

## Pins and audit scope

- Run: `I:/AnatomyTracker/runs/joint_v6_canonical_bridge_001`.
- Completion SHA256: `81257a308b0d6e10c70a53de72cafa932fd546c0ec2a6cd13aa82f1edfac3d2f`.
- Independent audit: `I:/AnatomyTracker/runs/joint_v6_canonical_bridge_001_independent_audit/audit.json`;
  SHA256 `d53664d3999054fe37c292b9ba080420384dbd499d235562dc4fdbb088462392`.
- A8000 checkpoint: `e67cfafbc2cc218053276734da0e56ec3ce7d4b3f37f36e3734b245f13e38504`.
- B8000 checkpoint: `c6aab521d86eee312f12327965b71c6da1d88f184b7e3427dd7ba2782c88c9f8`.
- B gallery: `f77e8e91f6f53249f0c4f2d0029d3db2a2e0f29a68a9c0d7124e0741a8b606f0`.
- Shared schedule: `314cd6579cdc9058fd0103e173878df7936e9f4d78ba0dc0bd2bd4bdacd75488`.
- Additional mode analysis: `I:/AnatomyTracker/runs/joint_v6_canonical_bridge_001_analysis/synthetic_mode_analysis.json`;
  SHA256 `5238ef86a5056052112558020bce6d96f0588ecbd5bbcc5fca90b29bf7998784`.
  Its exact source is preserved beside it as `analyze_bridge_modes_001.py`,
  SHA256 `fb6f786f9531f7d585495dc031433049319a48ed69210b9d87fce4be996cd128`.

The audit authenticated frozen bindings, both2000-row traces/shared schedules,
8000-step optimizer counters, canonical geometry/common eligibility and51 unchanged
nonretrieval tensors in each whole model. It independently reconstructed saved
two-reflection scores→normalised full-cell probabilities, stable top128, geometric
metrics, group means, parent gates and B−A differences. Maximum probability-log
reconstruction error was3.53e-6. It did not re-render, re-encode, replay gradients,
or independently replay every mined-negative selection. No calibrated probability,
animal-level final-validation or DeepSlice superiority claim follows from this run.
