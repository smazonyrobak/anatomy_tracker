# 139: blind-beam spatial atlas evidence and full-frame fit

## Question

Can anatomically coherent 2D-to-3D matches, learned from the actual frozen
132 blind-beam distribution, make an arbitrary-plane proposal more precise
and select it more reliably than the image-only prior? The 132/137/138
results leave a large gap between near-candidate availability and first
choice; 138's image-only anchor residual did not improve submillimetre
availability. This is a bounded mechanism test, not a GUI candidate.

039 already rendered exact continuous candidate frames/scales with the
section's finite PSF and robustly fitted a full plane. 083 already learned
candidate-centred coarse/fine atlas correlations. The 139 intervention is
**not** either mechanic alone: its candidates are only the real 132 blind
beam (no injected true or perturbed-true pose); hard wrong branches are
matched to near branches on atlas support; valid observed pixels supervise
an explicit unmatched/dustbin class, including partial sections; a
whole-slice contextual query and coarse-global/fine-local match jointly
drive a fitted full frame and a selected-branch score. The two matched
training arms differ only in atlas intensity, with atlas geometry/support
identical. Fixed-grid inference is the main pose/score path at TRAIN and
DEV; optional true-valid pixel queries supervise correspondence only and
cannot change the main fit or score.

## Frozen input and one-forward contract

Freeze the 132 batch-2,000 parent (`I:/AnatomyTracker/runs/v4_pose_adaptation_132`),
its image encoder, 16 base/64 antipodal-anchor states, mixture/reflection
scores, atlas matcher and tissue mapper. Reproduce the frozen blind16 beam:
top eight reflected base branches, top six anchor branches and two
antipodal-normal diversity anchors. Candidate state and reflection are
passed in explicitly. The new head sees only the input slice, frozen parent
features, each candidate's continuous full frame, and the Allen atlas.
For each branch it renders a finite-PSF seven-depth atlas neighbourhood
in that exact frame/scale, embeds atlas intensity and support, and compares
it with 4x4-global-context, 16x16-coarse and 32x32-fine image features.
The coarse match searches the full neighbourhood; fine local matches refine
the strongest coarse key. Both include a trainable dustbin. Matched 3D
points yield a two-pass robust, ridge-regularized affine plane fit. A
zero-initialized gate blends the fitted and parent physical O/U/V frames;
a separately zero-initialized score residual is added to the frozen pose
prior. Thus step zero reproduces parent corrected states and scores. No
truth, valid mask, target affine or smart-brush requirement enters the
inference path. This is one feedforward call with an analytic solve, not
an external iterative search. The local warp remains frozen in this gate;
its accuracy is evaluated only after pose selection is useful.

## TRAIN and controls

Use 2,000 optimizer updates with two independently drawn v4 and one v3
informative physical synthetic sections per update (6,000 accepted,
distinct physical sections total), one appearance each; do not make paired
recolours or reject chance-similar planes. Draw the same three sections and
candidate IDs once for both arms.
From its actual blind16 set, present at most four branches: the lowest
known valid-tissue rigid-error branch if within 1.5 mm, a wrong branch
at least 2 mm away with atlas-support fraction within 0.05 on the same
valid query sites, frozen prior top one, and a diverse branch if distinct.
The truth is used only for TRAIN branch selection and labels; no truth
candidate is inserted. If a near or support-matched wrong branch does not
exist, keep the section, train the available branches and record that fact.
Report how often matching is available. The support fraction is computed
before learning from the finite-PSF atlas support at the same sampled
observed-tissue positions, never from the image background.

The synthetic generator supplies `centre` (observed-valid-pixel-to-CCF
coordinates), `valid_mask`, rigid `state`, `reflection`, PSF offsets/weights
and full provenance. Sample 128–256 valid image pixels independently of the
fixed query grid, with the same sites for both arms. Their 3D key label is
the nearest supported rendered key when it is within 1.5 mm; otherwise
the dustbin. Removed or non-tissue pixels supervise visibility where
available but are not called anatomical mismatches. Coarse and fine
correspondence losses, robust fitted-plane displacement, and a listwise
selected-pose loss use the synthetic truth. The additional valid-pixel
queries are **auxiliary only**: fixed-grid logits/visibility always
produce the corrected state and score during training and inference.
If weak acquired coronal/sagittal TRAIN images are included, use their
inherited Allen affine only as a low-weight five-point pose-retention
constraint, never as expert dense correspondence truth; document donor IDs.

The treatment sees atlas intensity/support. The paired control sees the
same support, candidate coordinates, image, draw, optimizer schedule and
loss, but atlas intensity is zero. At evaluation, additionally report
source-image/feature swaps and within-support atlas-intensity shuffles;
these conditional ablations do not replace the matched support-only arm.
Train both new heads from random initialization, retaining all parent
weights frozen. Process each of the three images as a separate microbatch
with at most four branches, accumulate its gradient, then step once;
process paired arms sequentially. The 7x32x32 global key bank gives about
7,168 keys and 256 fixed queries, roughly 1.84 million coarse logits per
branch. Use candidate microbatches of at most four and mixed precision for
descriptors, FP32 for the geometric solve; one startup memory measurement
on the 11-GB RTX 2080 Ti is sufficient. Save step 0, 500, 1,000, 1,500
and 2,000 for recovery; terminal is the decision checkpoint, not a
DEV-selected checkpoint. Log physical
section and animal/specimen/experiment IDs, source/checkpoint/plan hashes,
and all attempted/accepted draws on I:.

## Frozen development decision

The first decisive readout is the frozen `fresh_v4_pose_dev_panel_132`:
248 eligible sections from eight reused synthetic DEV deformation plans,
with exact `target_centre_um`, `target_state`, valid mask and PSF. It is
heavily reused, so its predeclared gate is only a development decision,
not independent confirmation. Reproduce the 132/138 parent branch IDs
and pose errors before opening treatment results. Primary metrics are
mean 3D rigid-plane error on native-256 observed-valid tissue, not AP-only
error, for (a) blind16 truth-best corrected candidate ≤0.5 mm and
(b) blindly selected corrected candidate ≤1.5 mm and mean error. Report
fit-only versus prior-plus-fit selection, conditional near-to-precise
conversion, matched near/wrong pair wins, dustbin precision/recall, plan,
angle, support and raw/black/brush strata, and the v3 and acquired
donor-separated weak-reference guard panels used by 132.

To advance, treatment must improve blind16 ≤0.5-mm availability by at
least 15 percentage points over 132's 25/248, selected ≤1.5-mm capture
by at least ten points over 132's 73/248, and selected mean error by at
least 0.30 mm over 132's 3.282 mm. It must beat its support-only control
by at least ten points in selected ≤1.5-mm capture **and** 0.20 mm in
selected mean error, improve at least six of eight v4 plans, lose at
most five points on v3 near capture, and worsen no acquired DEV donor's
weak five-point affine discrepancy by more than 0.20 mm relative to
the frozen parent. The full arm's support-matched pair advantage must
remain positive when evaluated by physical pose error and collapse under
source swap or atlas-intensity shuffle; pair loss alone is insufficient.
All thresholds are simultaneous. Failure rejects this mechanism; do not
scale unchanged training or feed its fit loss into the direct pose head.

Any positive result must next replicate on new deformation-plan-disjoint
synthetic animals and independently referenced physical steep-oblique
sections before joint pose/deformation unfreezing. Neither this panel nor
weak Allen affines establish all-angle real-slide accuracy, calibrated
electrode-region probabilities, a GUI replacement or DeepSlice superiority.
