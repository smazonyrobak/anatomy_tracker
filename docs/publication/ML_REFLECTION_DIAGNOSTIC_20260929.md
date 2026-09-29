# ML-reflection diagnostic: a partial ambiguity, not a successful registration

Post-hoc CPU analysis of frozen curriculum003 step 20,000 and recovery004
step 10,000 supports approximate bilateral ambiguity as one contributor to
large MAP normal errors. It does **not** explain away poor capture. Original
absolute endpoints and failed scientific gates remain unchanged. No GPU work,
training, parameter changes, public benchmark or additional images were used.

For AP/DV/ML coordinates, `S = diag(1,1,-1)`. The diagnostic angle is
`acos(max(|n_truth·n_pred|, |n_truth·S·n_pred|))`, converted to degrees. It adds
brain ML reflection to the existing antipodal-normal equivalence. It does not
check plane offset, roll, tissue correspondence, hemisphere, or electrode sites.

| Normal-only diagnostic | Curriculum003 | Recovery004 |
| --- | ---: | ---: |
| MAP absolute mean error | 43.7265° | 47.6417° |
| MAP minimum over ML reflection | 27.4893° | 35.1850° |
| Absolute MAP errors >30° becoming <10° | 49/388 (12.63%) | 18/444 (4.05%) |
| Same rescued rows as fraction of all 640 | 7.66% | 2.81% |
| Best normal among top-32, absolute | 17.0658° | 27.7706° |
| Best normal among top-32, allowing ML reflection | 14.3091° | 20.4599° |
| Severe top-32 normal errors rescued, >30° to <10° | 10/125 (8.00%) | 40/254 (15.75%) |
| Normal probability mass within 10° of true normal | 7.0718% | 3.8412% |
| Normal probability mass within 10° of reflected normal | 6.9391% | 3.8446% |
| Mass within 10° of either normal, union without double counting | 13.0070% | 7.2786% |

Top-32 results are truth-selected, optimistic **normal-only** candidate-capture
diagnostics, not model-achieved alignment. Probabilities are uncalibrated and
marginalized over all offsets/rolls. The table uses the unchanged 640 sections
and 40 organizational synthetic groups, not independent biological animals.

Allowing a second target improves even unrelated predictions. A fixed-seed
100-shuffle descriptive control gives curriculum003 MAP errors 57.51°→46.27°
and recovery004 57.44°→46.25°. Thus approximately 11° of reduction occurs under
this unrelated-prediction control; the raw reduction is not itself proof of
intrinsic ambiguity. Curriculum003 top-32 errors under that control improve
35.33°→32.49°, similar in magnitude to its actual top-32 reflection reduction.

## Direct pinned-atlas symmetry measurement

The authenticated raw Allen NRRDs were decoded once with existing exact
preprocessing. Measurements compare normalized intensity and annotation/support
voxel arrays on CPU, in bounded AP blocks. Two geometrically specified ML
reflection centers were checked; neither was fitted to prediction errors.

| Reflection center | Intensity MAE within support union | Support Dice | Same integer annotation within support union |
| --- | ---: | ---: | ---: |
| 5687.5µm, voxel-extent midpoint (`ML index -> 455-index`) | 0.0145103 | 0.9976491 | 96.2054% |
| 5700µm, catalogue support-origin ML (`ML index -> 456-index`, zero outside) | 0.0104365 | 0.9965905 | 94.3955% |

The atlas is approximately, not exactly, symmetric. Same integer region labels
can occur in both hemispheres; this agreement does **not** preserve laterality.
Normals do not depend on the chosen reflection center, but reflected offsets
would. Offset endpoints were deliberately not transformed in this diagnostic.

Near-symmetry, nuisance image appearance and unknown raster reflection make
bilateral ambiguity plausible for some slices. These aggregate tests do not
prove any individual image is intrinsically unidentifiable: the residual atlas
asymmetry or additional subject/hemisphere information may resolve it. Most
severe errors remain unresolved, and curriculum003 still assigns about 87% of
normal probability mass outside both 10° cones. Learning/candidate capture
remains insufficient. Future modeling should preserve competing anatomical
modes and use genuine hemisphere/probe constraints; it must not silently merge
opposite anatomical locations or treat reflection as acceptable for surgery.

## Reproducibility

All files below are under `I:/AnatomyTracker/tmp/`. Original frozen arrays were
read-only. The normal-diagnostic JSON records their hashes, all mode/support
subsets, severe-error row indices and the shuffle control. Per-row results are
saved as `joint_v6_proposal_{curriculum_003,precision_recovery_004}_ml_reflection_diagnostic_rows.npz`.

| File | SHA-256 |
| --- | --- |
| `proposal_ml_reflection_diagnostic_20260929.py` | `1951d3fcd317358fd83cf83fca181e54d71a336868bb60f46ca708921ca40e1a` |
| `proposal_ml_reflection_diagnostic_20260929.json` | `f39003c824c4a2be3a24cd847e21c323c890c3adcc821487c2807ec7f58c8f32` |
| `allen_ml_symmetry_diagnostic_20260929.py` | `6b2e8a090ecbaab2765215a6e73baf57e97327ede2356d28e614d330c32c6b47` |
| `allen_ml_symmetry_diagnostic_20260929.json` | `c38baa79eb5f0368855a2119b2750525d8ee0e882be056b4d37995fc61560ade` |
