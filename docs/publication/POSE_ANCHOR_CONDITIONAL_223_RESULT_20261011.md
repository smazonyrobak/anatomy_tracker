# 223 matched-draw anchor ranking: no useful gain

The [pre-pinned protocol](POSE_ANCHOR_CONDITIONAL_223_PROTOCOL_20261011.md) tested whether an extra proper ranking loss *within* the 128 anchor/reflection branches fixes their poor ordering. Both arms continued the same project-random-initialized 221 framed checkpoint for 400 batches. The independent audit verified identical random seed and byte-identical draw logs: 1,239 attempts, 1,200 distinct accepted synthetic physical sections (800 new-geometry), 400 Lupori TRAIN presentations, and 200 each older coronal and sagittal weak-reference presentations per arm. Parent, protocol, split, 14 source files, config, draw/training logs, and 0/200/400 checkpoint hashes matched receipts. AL1A was excluded from gradients. The intervention's extra KL loss was active and positive on all 400 batches (mean 1.850); the control's was zero.

| TRAIN-inner panel at batch 400 | Control | Conditional ranking |
| --- | ---: | ---: |
| Fresh framed 64: highest-scored anchor has <2 mm **exact-affine** error | 9/64 | 9/64 |
| Fresh framed 64: some anchor has <2 mm exact-affine error | 41/64 | 41/64 |
| Fresh framed 64: global first-choice dense median | 3.097 mm | 3.096 mm |
| Fresh framed 64: best-of-160 dense median | 1.545 mm | 1.543 mm |
| Frozen framed 64: highest-scored anchor has <2 mm exact-affine error | 7/64 | 7/64 |
| Frozen framed 64: some anchor has <2 mm exact-affine error | 43/64 | 43/64 |
| Frozen framed 64: global first-choice dense median | 3.510 mm | 3.502 mm |
| Frozen framed 64: best-of-160 dense median | 1.554 mm | 1.555 mm |

The primary within-anchor gate **failed on both panels**. The highest-scored free branch still exceeded the highest-scored anchored branch by a median 2.45 versus 2.47 log units on fresh framed cases; neither arm selected an anchor as global top-1 on any of those 64 cases. This is not an anchor-coverage shortage alone: 41/64 fresh cases had a <2 mm anchored candidate, but only 9/64 ranked one highest within the anchor family. Global first-choice changes were at the scale of a few micrometres of the reported median, not a meaningful registration improvement. At 400 batches, this conditional KL did not solve the identified ordering or family-mass gap. It does **not** prove the broader joint pose-and-deformation architecture cannot work with more or better data.

The human-image guards did not show a large adverse effect: 52-section AL1A TRAIN weak five-point median was 1.312 mm control versus 1.302 mm intervention; older distinct TRAIN donor-equal weak five-point means were 0.580 versus 0.584 mm coronal and 1.346 versus 1.343 mm sagittal. AL1A and these donor guards use annotation-derived weak references, not physical arbitrary-oblique plane truth. The strict preregistered no-regression criterion also fails for the small coronal difference and frozen best-of-160 affine difference, but the absent within-anchor gain alone decides non-promotion. No checkpoint is selected for GUI deployment, uncertainty calibration, large-scale synthetic training, or DeepSlice comparison.

The next decision should not be another logit-loss tweak on these 64 cases. The model has to exploit candidate-specific anatomical evidence and a reliable local correspondence/fit signal, while improving synthetic appearance against acquired TRAIN slides and testing on genuinely new animal-level development data. A one-slice prediction can also remain intrinsically ambiguous when little tissue is visible; the eventual model must retain multiple plausible planes and abstain or widen uncertainty rather than force a confident wrong one.

Frozen code: commit `0d1e706`. Artifacts: `I:/AnatomyTracker/runs/pose_anchor_conditional_223_{control,intervention}_train_001`, `I:/AnatomyTracker/runs/pose_anchor_conditional_223_paired_readout_001` (224 cases, 1,568 rows; rows SHA-256 `2ebf6298a03afca643246edd7cea3c951c18a57de2593736b566bca07c129db6`), `I:/AnatomyTracker/runs/pose_anchor_conditional_223_AL1A_train_guard_001` (52 sections, 364 rows), and `I:/AnatomyTracker/runs/pose_anchor_conditional_223_old_train_guard_001` (88 sections, 616 rows). All evaluation output hashes and evaluator-source hashes were checked after exit with zero mismatches. No DEV/CAL/FINAL or public benchmark was accessed, and the probabilities remain uncalibrated.
