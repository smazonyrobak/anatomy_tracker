# Prepare-only raw Allen image-key diagnostic

`training/evaluate_joint_v6_imagekey_allen_raw.py` is deliberately locked by
`CHECKPOINT_SHA256`, `GALLERY_SHA256` and `AUDIT_SHA256`. Any `UNSET` binding raises
before any run-output read or model import. Image-key001 and its independent
audit have now exited successfully; the externally supplied bindings are:

- checkpoint4000: `d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5`
- gallery4000: `1568da8f1347641bba4ff93679ff39d5d6250693345db5d915322312bbe64666`
- independent audit: `f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b`

The script additionally requires the audit's integrity and performance gates to
pass. Do not substitute hashes from unverified files merely to bypass this lock.

Once approved, run with the I: environment and `-B -u -m
training.evaluate_joint_v6_imagekey_allen_raw`. It loads the whole step4000
checkpoint and its matching stored FP16 full gallery. Only the 64 frozen real
images from six development donors are newly encoded; no atlas volume is loaded
or re-rendered. No old model weights, externally learned features or pseudolabels
are imported. Checkpoint, gallery, audit, catalogue and inference-source hashes
are checked. The output is separate from the frozen experiment.

The existing raw-Allen evaluator is flat rather than a reusable function, so its
physical resampling block is copied without changing its mathematics: acquired
JPEG red/255, acquisition-centred 12-mm field at96², Gaussian antialiasing,
bilinear interpolation, acquired-border median padding, zero outline and zero
availability. There is no tissue crop, segmentation or tissue normalization.
Every source-pixel transform, image hash, original donor/specimen/experiment/
section ID, input raster and upstream affine is retained. The existing explicit
12.5-µm API-to-model voxel-centre conversion remains unchanged.

FP32 query/gallery-normalized cosine scores use temperature0.1 and apply each
cell/representation prior once. All98,304 cells and both raster-reflection keys
remain present; no training-time near-neighbour exclusions or truth injection
occur. Save full unnormalized component scores, normalized cell log probabilities,
query descriptors, stable top128 IDs and representation conditionals. Coarse
normal/offset errors and top32/128 plane capture use the upstream Allen affine
reference, with capture requiring the same candidate to satisfy normal<=10deg
and sign-aligned offset<=500µm. Sections are averaged within donor, then donors
equally. No new target catalogue label is fabricated.

These are uncalibrated coarse domain diagnostics, not blinded anatomical accuracy,
dense-deformation truth, reflection accuracy, trajectory confidence, an untouched
biological-generalization test or a DeepSlice benchmark. The cohort's historical
exposure caveat and near-coronal fluorescence sampling remain material. The
source is prepared and hash-bound only. Completed audit/configuration metadata
were inspected after explicit exit confirmation; no checkpoint/gallery encoding
or GPU execution has been performed for this preparation.
