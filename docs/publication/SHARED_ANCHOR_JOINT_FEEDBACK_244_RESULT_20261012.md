# Shared-anchor joint pose/warp 244: frozen development result

## Decision

**Do not scale, promote, calibrate, or benchmark this checkpoint.** The fresh,
randomly initialized 244 model improved its coarse pose proposal, but atlas
fitting did not reliably choose a better plane, pose feedback did not close the
blind selection gap, and the learned local warp was effectively the identity.
This is a development failure of the fitted-selection objective, not evidence
that arbitrary-plane registration is impossible. A targeted candidate-score
audit is the next experiment; another long run of the same objective is not.

The [protocol](SHARED_ANCHOR_JOINT_FEEDBACK_244_PROTOCOL_20261012.md) and
[checkpoint rule](SHARED_ANCHOR_JOINT_FEEDBACK_244_READOUT_RULE_20261012.md)
preceded the held-out readout. The rule's lowest-score *diagnostic* checkpoint
is batch 1,000. Batch 3,000 is the only checkpoint that passes its mechanical
eligibility gates, because intact atlas selection beats an atlas-null control
by **8.16 µm** on a **4.172 mm** error. That sign-only gate proved too permissive:
the effect is ~0.20% of error, is not stable at batch 6,000, and does not
justify a scientific claim of atlas-specific ranking. This interpretation
does not alter the frozen score or its receipt after seeing the result.

## Matched readout

The frozen panel has 32 synthetic DEV sections from eight synthetic
deformation plans and 12 acquired Allen DEV sections from six donors. Each
synthetic number below is the mean of section means within each plan, then the
mean of plans; acquired numbers are donor-equal means. Units are millimetres
of physical CCF site error except the acquired column, which is only distance
to an unverified weak affine. “Best of eight” uses truth to diagnose candidate
coverage and is **not** an output available at inference. Exact-pose mapping
also uses truth and tests only the local warp.

| Batch | Prior selected rigid | Best of eight rigid | Fitted selected rigid | Fitted selected mapped | Atlas-null selected rigid | Exact-pose mapped | Acquired weak: prior → refined |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.780 | 3.575 | 4.780 | 4.780 | 4.780 | 0.1603 | 12.017 → 12.017 |
| 1,000 | 4.171 | 3.145 | 4.171 | 4.172 | 4.171 | 0.1610 | 3.319 → 3.319 |
| 3,000 | 4.090 | 2.895 | 4.172 | 4.172 | 4.180 | 0.1602 | 2.910 → 3.974 |
| 6,000 | 3.864 | 2.815 | 4.037 | 4.037 | 3.887 | 0.1598 | 2.780 → 5.615 |

At batch 3,000, selection misses the best available candidate by **1.277
mm**, and fitting worsens its own top-one prior by 0.083 mm. At batch 6,000,
fitting worsens the prior by 0.173 mm, while the shifted-atlas *null* selects
0.149 mm better than the intact atlas. The selected branch changes from prior
top one on only 4/32 synthetic sections at batch 3,000 and 3/32 at batch
6,000. The refinement and candidate score therefore provide neither useful
correction nor dependable anatomical reranking. The acquired weak-label
comparison likewise worsens under fitting on five of six donors at both
3,000 and 6,000 batches; this is a retention warning, not biological truth.

The true-pose, no-warp baseline on these 32 synthetic sections is **0.160289
mm**. The learned exact-pose map reaches 0.159793 mm at batch 6,000, a gain
of only ~0.0005 mm. This supports “no meaningful local-warp improvement,”
not a claim that no deformation was present. A mapped error near 0.16 mm
when pose is supplied does not rescue the ~4 mm blind output.

The predeclared dimensionless checkpoint scores for batches 0/1,000/3,000/
6,000 are 1.000/0.753326/0.764371/0.769015. Thus batch 1,000 is the
diagnostic best but fails the atlas-specific-selection gate outright; the
single formal pass at 3,000 is quantitatively trivial. The protocol also
required a **meaningful** blind physical-error reduction and atlas-specific
fit advantage before scaling. Those requirements are unmet.

## What this does and does not diagnose

The coarse top-one pose improved across training, and the best of eight is
substantially better than what fitting selected. Candidate coverage is still
inadequate for high-quality registration, but it is not the only bottleneck.
The immediate failure is the candidate score/feedback stage: it cannot rank
nearer physical planes using the intact atlas better than the null control.
The true-pose local-map head also needs diagnosis rather than a claim of
effective deformation correction. A fixed-beam, per-candidate comparison of
score, physical error, render agreement and atlas-null margin can distinguish
a scoring/geometry bug from insufficient anatomical information.

At batch 3,000, selected rigid error is 4.050 mm with exact-black exteriors,
3.731 mm with imperfect brush, and 4.642 mm on raw backgrounds. For tissue
visibility below 15%, it is 4.735 mm; the 30–45° stratum is 4.579 mm.
Only one section in this panel is at 45–55°, so it cannot establish
heavy-oblique performance. These are descriptive, small-panel strata, not
independent gates. The v7 generator spans all plane normals and varied
backgrounds, tears, fragments, folds, bubbles, seams and local strain, but
its tissue texture is still atlas-like and its artifacts are uncalibrated
against a representative acquired distribution. All synthetic plans are
derived from one atlas, **not** eight biological animals. The acquired labels
are mostly coronal weak affines, not independently measured 3D physical pose.
No arbitrary-oblique animal-level accuracy or calibrated electrode-region
probability is established here.

## Integrity and next step

Training completed 6,000 batches with 12,000 independently drawn synthetic
sections and 6,000 distinct acquired TRAIN sections. The model has 18,161,735
randomly initialized parameters. The independent verifier reported
`verified: true` for four checkpoints and 176 readout rows. The train and
readout completion-receipt SHA-256 values are respectively
`ed6d3b8480b2df437c61739eff036f392a31f67f1f7143962d5bb46f1562952f`
and `ee7505ecc9f2f56655880f9d17c46292729ab4dd4b3a87e36d77ddd32c4c3694`.
Raw row and summary SHA-256 values are
`d69eaf510f36afa78921ca47329d3c3b4e3d015b1a62ae006e4df50db5a3c26d`
and `0dda40c0c8e17e0262df3656fab9a7008fbac928021fb6d4bcf91d0ee92a0ffd`.
The frozen output roots are
`I:\AnatomyTracker\runs\shared_anchor_joint_feedback_244_train_001` and
`I:\AnatomyTracker\runs\shared_anchor_joint_feedback_244_readout_001`.
The checkpoint receipt records `calibrated: false` and
`public_benchmark_used: false`; no CAL/FINAL animals were used.

Next, freeze a narrow per-candidate diagnostic on the same development
panel, inspect both correct/nearby/wrong candidate render matches and the
fitted score's correlation with known physical error, and repeat the ranking
against atlas-null. Correct a demonstrated geometry/score fault or make a
small matched architecture change only after that diagnostic. Keep the
one-model, any-plane requirement, optional acquisition/smart-brush evidence,
specimen provenance and animal-separated calibration/final sets. Do not
silently select the truth-best branch, train on the panel, or launch the
public comparison to DeepSlice.
