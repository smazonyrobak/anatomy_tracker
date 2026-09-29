# Train-only anchor/gallery closure diagnosis

Prepared, not executed. Driver: `training/diagnose_joint_v6_real_training_retrieval.py`.
Output: `I:/AnatomyTracker/runs/joint_v6_real_training_retrieval_001`.
This is one inference-only diagnosis, not another training run or a promotion gate.

Compare the complete, separately loaded whole A6000 and failed real+synthetic8000
checkpoints on all 256 unchanged red-channel, acquisition-centred 12mm/96² inputs
from the 58 training donors. No six development donors, alternative channels,
preprocessing tuning, optimisation, encoder merging or benchmark data are used.
The source/inputs/checkpoints/galleries and completed integrity audits are pinned.

Render all 256 recorded continuous upstream-affine anchors once with the recorded
50µm nine-point PSF, then encode those same intensity/support arrays under each
whole model. This PSF is an engineering assumption, not measured optical truth.
Reuse each endpoint's saved full 98,304-cell/two-reflection gallery unchanged;
do not rebuild or repair that gallery. Preserve every anchor, including the 11
previously observed zero-support anchors and one additional support<64 anchor.
Report actual newly rendered support and intensity separately: absent annotation
support does not logically establish a zero intensity image or an affine bug.

Readouts are own-anchor rank among all 256 anchors, best near-equivalent anchor
rank, explicitly secondary incompatible-only rank, and appearance margins.
Near equivalence uses the training rule: antipodal normal≤10° and finite-frame
four-corner RMS≤1mm, minimum identity/horizontal raster correspondence with the
actual 0..95 pixel-centre edges. The unmasked full catalogue posterior uses its
representation priors and cell masses once. Its MAP errors and top1/32/128
physical-plane capture use normal≤10° and sign-aligned normal offset≤500µm;
this is distinct from finite-frame equivalence. Save nearest finite-frame cell
error/rank, equivalent/plane-eligible candidate counts and best eligible rank.
Across-space margins use uniform-reflection appearance scores without cell mass,
not separately normalised posterior values. Ties favour the lower index.

Persist full cell log probabilities, query/exact-anchor descriptors, all exact
anchor raw component scores, priors, reference geometry, exact row IDs, and input
hashes sufficient to reconstruct catalogue logits from the frozen banks.
Provide donor means and equal-donor macros for all rows, zero support, positive
support<64, and support≥64. Missing eligible candidates remain missing with finite
row counts; they are never silently replaced or filtered. Fixed grayscale previews
show every support<64 query/anchor intensity/support plus the first 12 remaining
rows, without intensity normalization or cherry-picking.

Predeclared interpretations:

- Good exact-anchor retrieval but poor full-gallery capture plus a large nearest
  finite-frame mismatch supports an anchor/catalogue representation gap.
- Poor all-anchor retrieval despite the low sampled training NCE supports a
  sampled-negative/objective limitation or content mismatch; empty-support strata
  help localise that failure but do not prove a coordinate defect.
- Good training full-gallery capture with the already frozen poor development
  result supports a donor/domain generalisation gap, not biological accuracy.
- Improving one of these training readouts does not override the failed frozen
  development gate or justify reusing development donors for tuning.

No numeric outcome is assumed. These upstream affine references are weak geometric
diagnostics, not dense histological correspondence, deformation, uncertainty, or
independent biological ground truth.
