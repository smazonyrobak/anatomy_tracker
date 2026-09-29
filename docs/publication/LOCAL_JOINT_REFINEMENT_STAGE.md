# Conditional local joint-refinement stage

`training/run_joint_v6_local_refinement.py` trains the existing single v6 model's
shared recurrent pose updater and affine-free SVF decoder directly. It bypasses
the coarse cascade **only for explicitly truth-near, teacher-initialized local
learning**. This is not a substitute for global capture, an end-to-end benchmark,
or a claim that the complete system works. No training has been launched by the
script's author; the parent checkpoint constant is provisional pending review.

The complete own-lineage `joint_v6_proposal_curriculum_003/step_20000` checkpoint
is the current provisional driver constant, not a final parent selection. The
next local run must continue the selected **whole** global-model checkpoint;
independently trained global/local encoders will not be merged. Proposal
parameters, the histology stem, and the shared
encoder stay frozen; otherwise a locally updated image encoder would silently
change the trained global proposal. Trainable modules are the atlas stem,
refinement pair encoder, GRU, nine-coordinate update, representation likelihood,
and deformation decoder. Outputs remain one full-model checkpoint, not a second
independent deployed model. The final frozen-parameter equality check runs once.

The authenticated tensor pack contains all 2,048 training and 256 development
joint-component rows, including censored observations, with exact lineage and
original cache/row receipts. Synthetic animal IDs denote organizational groups
from one Allen atlas, not independent biological brains. Raw, black-exterior and
imperfect-brush input semantics are unchanged; segmentation is not inferred.

## Fixed experiment

- 4,000 steps, batch four, fresh AdamW at `2e-4`, weight decay `1e-4`, gradient
  norm cap one; FP32 by default. Global weights are retained, not retrained here.
- Three recurrent pose updates and the original pose-only prefix of one.
  Deformation integration uses seven scaling/squaring steps to match frozen
  target certification; the parent's proposal-only setting of three never trained
  this decoder. Target amplitudes are within the decoder's existing bounds.
- One local cell, two exact identity/horizontal raster representations, uniform
  prior. Reflection truth is used for cross-entropy only, never supplied as a
  one-hot prior or used to select recurrent context. Reflected SVF/map targets
  are already expressed in the observed raster and are not flipped again.
- Per-row original symmetric nine-sample PSF, at the original 96x96 canvas.
  Exact synthetic PSF knowledge is an explicit condition of this experiment.
- Frozen shuffle and perturbation tensors, saved before learning. The first
  500 steps use uniform coordinate bounds
  `[.06,.06,250,.08,300,300,.05,.05,.05]`; remaining steps use
  `[.15,.15,600,.20,600,600,.12,.12,.12]`. Their order is normal tangent u/v
  (radians), normal offset (micrometres), roll, in-plane translations u/v,
  log-basis u/v, and shear. This is **not** the SO(3)/translation ordering of
  `full_frame_residual`.

The objective contains discounted full nine-coordinate point residuals, physical
frame landmarks, reflection CE, and the existing SVF/map/support/topology/
smoothness/inverse-consistency losses. Dense losses use the original valid,
non-abstaining correspondence weights and the product of pose and dense row
eligibility. Prefix identity fields are excluded from deformation supervision.
The new correlated covariance head remains disabled until point learning works.

## Local-development boundary

Development starts are frozen independently at the larger perturbation range.
The zero-step and final evaluations use all 256 rows; intermediate evaluations
every 250 steps use the same fixed 64. Deformation feedback is enabled for all
development rows: truth censor flags determine which metrics are identifiable,
not the model's predictions. Recorded metrics include initial/final canonical
landmarks, rotation, reflection classification, SVF and map errors, Jacobian
minimum, and joint CCF correspondence error using the **predicted** reflection.
Raw frames, maps, velocities, reflection probabilities and row indices are saved.

Full-model/optimizer/RNG checkpoints, source hashes, parent hash, pack hashes,
row identities, schedule, loss trace and per-group metrics make the local result
replayable. Group-macro metrics exclude unavailable targets and retain eligibility
counts. Improvements establish conditional local-capture/deformation learning
only. Later honest global evaluation must use image-selected starts, no truth
catalogue index or truth deformation gate, and only observable acquisition
metadata. The remaining common deterministic frame/warp across representations,
calibration, constraint/ray inference and desktop delivery are not solved here.

## Completed convention review — 2026-09-29

The driver, packed targets, native `refine()` and v6 losses were reviewed with
no launch-blocking convention bug found and no driver change required:

- Full-frame loss uses local SO(3), translation and basis/shear coordinates;
  initialization correctly uses the distinct plane-tangent perturbation order.
- Three pose updates yield four deformation states, with activity
  `[false, true, true, true]`. The identity prefix is excluded from dense loss.
  Pose states, recurrent hidden states and SVF feedback maps retain gradients
  across re-rendering. Only representation-mixture probabilities entering the
  deformation context are detached, not the context or deformation itself.
- Dense supervision uses pose eligibility times dense eligibility, including
  correct exclusion of the one training row with pose weight zero but dense
  weight one. Support BCE targets the original `deformation_weight`, matching
  the native loss rather than substituting a tissue-segmentation target.
- Source SVFs and pullbacks are already conjugated into the observed reflected
  raster. Dense CCF composition correctly applies the selected raster reflection
  **after** the pullback, then physical OUV with `x/96` and `y/96`; no additional
  half-pixel shift or second SVF conjugation belongs here.

One bounded CPU-only recomposition of packed truth on all 231 eligible development
rows confirmed the last point: mean row-weighted CCF discrepancy was
`0.000664765 µm`, maximum row-weighted discrepancy `0.001327075 µm`, and maximum
supported-pixel discrepancy `0.003300348 µm`. Identity/horizontal mean discrepancies
were `0.000676986 / 0.000651552 µm`. This verifies coordinate agreement, not learned
accuracy. No GPU training was launched for this review and active global-training
outputs were not accessed.

The first stage is a frozen-source-encoder proof of conditional truth-near local
learning with known PSF, not global-capture or uncertainty qualification. A later
joint-unfreezing stage must retain proposal rehearsal within the same whole-model
lineage; it is not implemented or qualified by this review.
