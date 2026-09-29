# Joint source-feature adaptation with broad proposal replay

Prepared on 2026-09-29; not launched or qualified by this commit.
Driver: `training/run_joint_v6_joint_rehearsal.py`.
Output: `I:/AnatomyTracker/runs/joint_v6_joint_rehearsal_003`.

## Matched local protocol

Continue the **whole** `joint_v6_proposal_curriculum_003/joint_model_step_20000.pt`,
not the independently trained local002 endpoint. The coordinate-evidence hook
is ON and zero-initialized; metadata conditioning and covariance remain OFF.
Keep local002's seed2026092906, 4000 steps, local batch4, first500 small pose
perturbations then3500 larger perturbations, fixed development perturbations,
three refinement updates, one pose-only step, and the same geometric losses.
The local schedule and `local_pass` source are unchanged. FP32 AdamW remains
LR2e-4, weight decay1e-4, and one aggregate gradient clip at1 per update.

The intervention unfreezes the histology stem, shared encoder, any existing
spatial residual blocks, and full-catalogue proposal head, in addition to the
local002 trainable components. Shared-encoder adaptation affects the atlas
branch too. It is not a source-exclusive or coordinate-effect experiment.
Accumulate the original local loss backward, then **1.0 times weighted
full98304-cell proposal NLL** backward, before one optimizer step. No auxiliary
normal NLL,008 kernel score, independently trained encoder, or external weights
are introduced. Pose-anchor rows receive proposal supervision only, never
fabricated zero-deformation targets.

## Broad replay and exact provenance

Reuse completed003's first4000 mixed batches:8 rendered observations plus8
frozen rows each. The original generated observation IDs, appearance/noise
seeds, horizontal reflection, selected brush mode, thickness, and catalogue
indices are unchanged. The full schedules are authenticated and copied:

- `generated_schedule.npz`: `5f88b717d022026cc386f337e8979cfba11a25f4c491dfe05d49b53113678712`
- `frozen_training_row_indices.npy`: `ebef1776a96c83678c8d3e0c5b96590d40969ffed1cd392a3f65b488fb37ee66`
- Original generator snapshot: `550bb4660fbfbc10752f23442c02c78e3bb6f0282667a24a597ff572e4461705`

These are **32000 replayed generated observations and32000 frozen-row
presentations, zero new observations**, not new biological subjects. The
frozen5120-row pool includes3072 pose anchors and2048 complex joint rows;
same-local-row rehearsal alone would omit those pose anchors. The generated
prefix has32000 distinct cells from the complete-catalogue permutation, not
new complete98304-cell coverage. Rendering retains the original normalized
nine-sample finite-thickness PSF and all three modes: raw synthetic background,
exact black exterior, and imperfect brush. Finite-slab and visible support
must both reach64 pixels for point supervision; censored observations are not
resampled away. No automatic segmentation or real acquired-background claim.

At launch the script authenticates the compact2048/256 local packs and
5120/640 proposal tensors once, using their frozen receipts and original003
prepared hashes. It records the full parent hash, catalogue receipt/hash,
source hashes/git state, local schedule hash, replay schedule hashes, exact
IDs for every pool, and copied original schedules/generator. Training inputs
are images and brush channels; target labels derive from synthetic geometry,
not prior learned features or pseudolabels. The separate appearance-noise
variable cannot overwrite the fixed local pose-noise schedule.

## Readouts and interpretation

At steps0/4000, evaluate all256 local and640 proposal development rows.
Intermediate local evaluation remains the fixed64 rows every250 steps.
Save local raw states/maps/reflection probabilities, full98304-way raw proposal
log probabilities and cell predictions, exact rank/topK and plane/frame/offset
errors, and organizational-group, input-mode, and censor-stratum summaries.
Each periodic checkpoint contains the same whole model, optimizer, RNGs and
schedule references; endpoint checkpoints also contain proposal summaries.

Compare source adaptation plus replay against local002 while keeping the
coordinate choice fixed. This does not separately identify the effect of
replay versus unfreezing. Local starts remain truth-near with known synthetic
PSF, so success is conditional local learning, not honest global registration.
The broad proposal readout measures preservation of retrieval separately.
Compare proposal retention against this run's own FP32 step0 endpoint, not
historical AMP numbers that would confound precision with parameter changes.
Synthetic group separation is organizational, not independent biological
generalization; no calibration, electrode-location probability, public
benchmark, DeepSlice superiority, or shipping claim follows.

Only syntax and source-level protocol review are performed before launch;
the driver is not imported or executed during preparation.
