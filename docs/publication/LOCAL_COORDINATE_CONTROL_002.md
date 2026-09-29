# Matched local coordinate-evidence control 002

Prepared 2026-09-29 while local001 is active; not launched or selected yet.
Proceed only after its confirmed exit and independent endpoint audit. This is
a targeted local-learning control, not global localization or benchmarking.

The sole intended model change from local001 is
`coordinate_evidence_conditioning=True`: seven predicted geometric channels
enter the shared nonlinear GRU through 896 zero-initialized weights. See
[coordinate implementation](RECURRENT_COORDINATE_EVIDENCE_20260929.md).
The hypothesis is that explicit spatial/frame/reflection information will make
the pooled pose update more learnable. It is not an established cause of poor
learning; convolutional boundaries can also provide implicit position cues.

Restart from the identical **whole** own-lineage 003 step-20000 parent, not
local001's trained endpoint. Keep every existing parent tensor exactly, add only
`pose_model.coordinate_evidence.weight`, and use a fresh optimizer. Include that
new projection among trainable parameters. Keep the source encoder and global
proposal frozen, optional soft-offset inputs and joint covariance disabled.

Match local001's 4,000 steps, batch four, seed, row order, perturbations, FP32,
optimizer, losses, clipping, three recurrent steps, pose-only prefix, known PSF,
reflection prior, frozen data and eligibility. Confirm saved schedule tensors
match when auditing the completed run. Initial common model parameters and
zero-conditioning evidence were already checked exactly; avoid another broad
compatibility suite. Preserve full checkpoint/source/data/raw prediction receipts.

Use the same predefined absolute continuation gate as
[local001](LOCAL_JOINT_REFINEMENT_STAGE.md): at least 20% paired reduction in both
landmarks and dense CCF, at least 10% map-error reduction from identity, no worse
normal error and no nonpositive valid-tissue Jacobians. Also report the paired
002-minus-001 endpoint differences, by brush mode and true reflection. A partial
improvement may identify useful evidence, but does not satisfy this gate.

Do not change loss weights in this control. The current approximately 0.27-pixel
target deformations are small compared with starting pose error; their loss scale
is a separate hypothesis, not a demonstrated bug. Combining a weight change here
would confound the coordinate-input comparison. No claim of animal generalization,
calibrated uncertainty, real-histology accuracy or deployment readiness follows
from this one-atlas, truth-near experiment.
