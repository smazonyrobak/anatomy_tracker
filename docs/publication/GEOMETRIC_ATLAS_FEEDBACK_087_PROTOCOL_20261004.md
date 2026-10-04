# 087: fixed geometric feedback experiment (frozen before run)

## Question

The 086 audit found weak rank evidence from scalar fitting scores. Can the already-trained 083 **spatial correspondence posterior** itself improve the predicted plane by a geometry-consistent update, without retraining, changing the beam, or using synthetic truth at inference? This is one targeted mechanism test, not a candidate for GUI promotion.

## Frozen change

Load the 085 batch-1,000 model and head weights unchanged. In a new subclass, use 083's existing depth–dy–dx posterior over atlas cells. On pass one, use the coarse 16×16 posterior; on pass two, re-render at the new plane and use the fine 32×32 posterior. Convert each pixel's posterior mean into a local 3-D residual in millimetres, accounting for physical in-plane basis, reflection and through-plane offset. Solve one confidence-weighted 6-DOF rigid least-squares update across the whole slice with Jacobian [minus-skew(x), I]. Confidence is matched atlas support times one minus normalized posterior entropy; add 0.05 × height × width × I ridge, take a half-step, clip each rotation component to ±0.35 rad and each translation to ±3 mm. Do not update the in-plane basis or nonrigid map from this solve. Preserve the frozen mapper and fitted score, which may be distribution-shifted by the new poses. No new trainable parameters, threshold search, or use of ground truth in prediction.

The operation is inspired by [geometric consensus in multimodal registration](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/el.2018.6713) and [single-slice/volume feature correspondence](https://doi.org/10.1007/978-3-031-73480-9_22). These studies do not establish that this fixed update works for torn, variably stained mouse histology; that is the empirical question here.

## Matched readout and interpretation

Use the same 246 eligible synthetic DEV sections/eight held-out deformation plans, same 14 blind 085 branches, same fixed 1024 surviving pixels for all-candidate 96-grid comparisons, and all surviving pixels for the selected 256-grid map. Require each beam to equal the frozen 086 branch list. Pair every row with the frozen 085 batch-1,000 result, retain section/animal/specimen/experiment and physical-plan IDs, report every candidate's score and physical error, and aggregate first within plan then equally over plans. Report selected 96/256 error and truth-best-of-the-*same*-14 96-grid error, overall and by appearance. Do not open images, real calibration/final animals, or public benchmark material. No checkpoints are trained or selected.

An improvement in best-of-14 error without improvement in score-selected error would support geometric capture but identify ranking as still missing; it would **not** establish a deployable model. A best-of-14 non-improvement or deterioration would argue that current 083 correspondences lack sufficiently accurate spatial structure for this update; do not merely turn up its gain. The primary comparison is paired 085→087, with parent 059 retained as context. This synthetic-development panel cannot establish biological accuracy or calibrated uncertainty.

Inputs remain frozen under the 085 training/evaluation receipts and 086 candidate receipt. New code and outputs live only on I:; the 087 evaluator records input/source hashes and raw rows with a completion receipt.
