# 187: fluorescence pose-capture inner-animal result

The [precommitted protocol](LUPORI_FLUORESCENCE_POSE_ADAPTATION_187_PROTOCOL_20261011.md) trained the same 132 scratch-lineage joint model for 1,200 batches with three independently sampled full-sphere synthetic slices, one Lupori acquired TRAIN slice, and one alternating older weak-real slice per batch. The image input used image-wide p99 normalization, with no tissue mask at inference. A mask was used only to sample supervised tissue points. `AL1A` was sealed from gradient updates as the inner TRAIN-animal guard; the official DEV/CAL/FINAL Lupori animals remain unread. All files are on I:.

The runner exited 0. An independent post-exit audit matched SHA-256 for the config, draw log, training log and four checkpoints; it counted 1,200 updates, 3,600 distinct used synthetic physical sections, 1,200 acquired presentations of exactly `CC2B,CC4B,CC4A`, and 1,200 old weak-real presentations. The runner receipt is `I:/AnatomyTracker/runs/lupori_fluorescence_pose_187_inner_AL1A/completed.json`; the selected step-800 checkpoint SHA-256 is `1152812f0986769a1f2ac922a1c9c2346436f685da2f5c7357d9b5d1ab31b4fc`.

Exact input-matched five-frame-point pose error on all 52 valid `AL1A` sections, against human VisuAlign planes (mm):

| checkpoint | first-choice median | first-choice ≤1.5 | all-160 oracle ≤1.5 | first-choice >5 |
|---|---:|---:|---:|---:|
| 132 parent / 187 step 0 | 7.198 | 1/52 | 3/52 | 46/52 |
| 187 step 400 | 1.647 | 17/52 | 38/52 | 3/52 |
| **187 step 800** | **1.344** | **28/52** | **39/52** | **2/52** |
| 187 step 1200 | 1.374 | 29/52 | 39/52 | 2/52 |

The predeclared inner gate passed by a wide margin at step 800: 81.3% lower median first-choice error versus parent, and 39/52 all-candidate capture versus the required 12/52. The all-candidate number is an oracle diagnostic, not an available inference result. The step-800 first-choice 90th-percentile error remained 3.82 mm.

Matched frozen retention checks at step 800: v3 synthetic mean direct-pose tissue error rose 1.1% and v4 rose 1.5% (both within the 10% ceiling). Older coronal and sagittal acquired donor-equal mean disagreement with inherited weak affines rose 0.081 and 0.018 mm. Step 400 had larger old-real regressions (0.315 and 0.339 mm); step 1200 had 0.162 and 0.139 mm. The step-800 trade-off is best on the predeclared inner guard while preserving the older domains most closely. The retention results are not expert-reference accuracy.

Raw inner-guard rows are `I:/AnatomyTracker/data/lupori_joint_185/inner_guard_candidate_187.json` (SHA-256 `c0fdd630b2450975a48cb726d3b2e5c5f2ca6c54fbab5192ab88e1c788d8d2d4`); exact parent rows are `inner_guard_parent_187.json` (SHA-256 `140999ebdd31fb3081a221ea39ab8706e36be9e23b520e1343c9532037e49e0c`). Frozen synthetic/old-real retention is `I:/AnatomyTracker/runs/lupori_fluorescence_pose_187_retention_eval/summary.json` (SHA-256 `8d90c3c384dfd6417a35fcd246b55a9d760ed5b7cc5dc75ed57cbe2cd35da1f4`); its completion hashes were independently verified. The evaluator explicitly allowed two documented historical code changes: an unused opt-in map derivative bound and a newer v3 artifact placement sampler; the parent direct `predict()` semantics and the frozen panel files were unchanged.

This is a strong *near-coronal fluorescence* pose-capture result on **one held-out TRAIN animal with approximate human alignment references**, not evidence of accurate electrode-region assignment, biological arbitrary-plane generalization, calibrated probabilities, fitting-to-pose feedback, or GUI readiness. The observed planes span only about 4.24° median / 11.12° maximum from coronal; synthetic full-sphere retention is not a substitute for acquired steep-oblique validation. Next: train on all four Lupori TRAIN animals under a new frozen run, open the single official DEV animal once to select among predeclared milestones, then address joint anatomical fitting feedback. Keep CAL and FINAL untouched.
