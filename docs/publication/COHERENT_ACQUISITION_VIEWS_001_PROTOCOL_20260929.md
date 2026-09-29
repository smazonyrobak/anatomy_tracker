# Coherent acquisition views 001 — frozen generation protocol

The flat generator `training/prepare_joint_v6_coherent_acquisition_views.py`
passed root and independent source review; ready for committed generation.
No model execution or live signed-pose output access is involved. Output:
`I:/AnatomyTracker/data/joint_v6_coherent_acquisition_views_001`.
Generate **128 new physical sections, 16 per existing TRAIN subject, and three
paired presentations per section: 384 observations**. Reuse the eight accepted
training plans; generate no development subjects, sections, or nuisance statistics.
This adds acquisition-frame variation to the existing full-box views; it does
not replace them or authorize a training launch.

## Frozen inputs and train-only nuisance law

- Subject parent: `I:/AnatomyTracker/data/joint_v6_coherent_subject_plans_002`,
  completed receipt SHA256
  `83fa8fc0ab19f15c8b64babdfe1e98251a972c6759dade08e5513e70c76a7c89`.
  Select only its eight `train` records, retaining every plan/animal/specimen/
  experiment ID, plan content receipt, accepted coefficients, scale and flow steps.
- Allen geometry: `I:/AnatomyTracker/data/allen_real_training_inputs_20260929/`
  `image_geometry.jsonl`, SHA256
  `9a718b2fb2791584e1d225123facbe1c02c96cfa701a9e6f3fc68f17a565d218`;
  its `summary.json` SHA256 is
  `4f8484bbd2ab5719971d9862f3f5ea87ea3d1c9de76468a936af67bcc65da9fd`.
  Use the 256 recorded model-pixel affines from 58 TRAIN donors only. No images,
  six-donor development geometry, model predictions, or benchmark data are needed.
- Reuse the authenticated v6 Allen atlas/support identity and the catalogue's
  declared `support_geometry.support_origin_ap_dv_ml_um`, denoted S. Record its
  exact numerical value and catalogue/source hashes before generation; do not
  substitute the array origin, bregma, tissue centroid, or full-volume box centre.
  The existing plan context SHA256 is
  `c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8`.
  Exact S is `[6600.0,3862.5,5700.0]` micrometres; catalogue
  `I:/AnatomyTracker/runs/joint_v6_proposal_substantive_001/catalogue.pt`
  SHA256 `9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71`.

For each recorded 3x3 affine A mapping model pixel (x,y,1) to physical AP/DV/ML,
set O=A[:,2], U=96*A[:,0], V=96*A[:,1]. The stored half-voxel conversion is already
included: add no further 12.5 micrometres. Define

    u = U / |U|;  Lv = |V - u*(u·V)|;  v = (V - u*(u·V)) / Lv
    c = O + (U+V)/2;  Lu = |U|;  h = (u·V)/Lv
    nuisance = (u·(c-S), v·(c-S), Lu, Lv, h)

Here h is dimensionless shear; V=Lv*(v+h*u), so Lv is not |V|. Keep the entire
five-value tuple from one row together, retaining correlations and its exact
Allen donor/specimen/experiment/section and geometry-row identity. Sample an
empirical donor-balanced distribution, not independent Gaussian fits or fitted
clipped ranges: concatenate fresh permutations of the 58 donors to 128 slots,
then select one recorded row uniformly within that donor. Assign slots in fixed
subject-index/section-index order. No donor becomes a synthetic biological parent:
its ID is explicitly the **acquisition-nuisance source**, separate from synthetic
animal lineage. Do not estimate normal, roll, plane offset, thickness, appearance,
or anatomical-deformation distributions from these near-coronal affines.

Proposed root seed is 2026092925, frozen before generation. Use independent PCG64
SeedSequence branches for donor schedule, row choice, and each subject/section's
plane/thickness, reflection, appearance and CPU noise. Save every seed and selected
row, not only the root. No resampling based on tissue visibility or model success.

## Preserve all-plane sampling; replace only the acquisition chart

Call existing `sample_subject_planes` once per section, at 96x96 with that
section's independently uniform 25–100 micrometre thickness, nine offsets and
the existing globally normalized masses [1,2,2,2,2,2,2,2,1]. Keep its uniform RP2
normal n, independent roll basis (u_s,v_s), and conditional conservative-box/slab
offset. Discard only its bounding-box-covering U/V lengths. The original plane
contains its saved origin p0; its full sampling record remains provenance.

Let F_j be the frozen accepted CCF-to-subject map. Compute S_j=F_j(S) once per
subject using the accepted forward flow, and project it onto the sampled plane:

    a = S_j + n * (n·(p0-S_j))
    c_new = a + nuisance[0]*u_s + nuisance[1]*v_s
    U_new = Lu*u_s;  V_new = Lv*(v_s+h*u_s)
    O_new = c_new - (U_new+V_new)/2

The new chart has exactly the sampled plane and roll, but independently borrowed
finite-FOV centre/span/shear. Both source and new chart use x/96,y/96: c is the
full-extent frame centre, whereas the actual 0..95 pixel centroid is
c-(U+V)/192. Save both explicitly; do not silently exchange them or multiply
the borrowed 96-extent basis lengths by 96/95.

**Scale convention:** F_j(x)=frozen_centre + G*(flow(x)-frozen_centre), where G is
the accepted positive AP/DV/ML diagonal scale, not necessarily isotropic.
The inverse first divides by G, then integrates the negative flow. Borrowed
shifts and lengths are used as physical subject micrometres without another G
factor; all anatomy scale variation remains in the frozen map. Applying G to
the new basis vectors would also change their sampled normal/roll. This explicit
engineering transfer of upstream CCF-chart dimensions is not a measurement of
true acquisition-FOV sizes or a claim that nuisance is biologically independent
of subject size. S_j uses the complete forward map, not global scale alone.

## Fresh physical rendering, truth and outside-domain rules

Pass the new OUV to `make_coherent_subject_section_v6`, with identity observed
2D processing and one independently sampled finite horizontal reflection.
Composition is observed pixel -> identity processing -> finite raster reflection
-> subject plane + z*n -> accepted subject-to-CCF map. Reflect spatial locations
once; do not flip physical vector components or the PSF direction. No stored
image crop, unobserved-raster padding, copied tissue labels, or old total-SVF
truth is used. Render the pinned atlas at the newly mapped coordinates.

Retain exact stored-numerical canonical centre and observed centre/slab CCF
coordinates, subject queries, PSF, and separate processing truth. Recompute both
full-canvas least-squares plane gauges and full 3D residuals with the adapter;
changing the finite chart of a curved surface generally changes its fitted
plane. Do not reuse the old fitted pose, discard normal residuals, or invent
nearest-catalogue-cell truth. Exact means the accepted finite-step FP64 subject
mapping followed by the existing FP32 atlas interpolation, not analytic anatomy.

Source review establishes the applicable exterior semantics:

- `arbitrary_plane_subject_deformation_v2.py` evaluates zero-extended cubic
  B-spline velocity coefficients. It does not clamp physical queries to a
  lattice edge. Beyond velocity support, the inverse retains the global inverse
  scale; it is not generally the identity map. Use the accepted RK4 steps and
  retain the original plan's audit scope, not an invented whole-space numerical
  certification. The opt-in FP64 Torch mapper has the same scale/flow ordering.
- The completed cohort's `render_finite_thickness_coordinate_grid` uses physical
  voxel centres origin+(index+0.5)*spacing, align_corners=True trilinear sampling,
  zero atlas padding and one global PSF normalization. No per-pixel tissue
  renormalization or coordinate clipping. Partial-volume boundary interpolation
  remains the existing renderer's convention. Annotation sampling is nearest,
  ties-to-even, with out-of-range labels zero.
- A finite chart may extend beyond the conservative subject box or miss tissue
  entirely. Keep it. Recompute finite/visible support and annotation/PSF weights
  at its new coordinates, retaining eligibility finite_mass>=64 and
  visible_mass>=64 for each presentation. Record out-of-atlas and low-information
  counts; do not retry, replace, or constrain the geometric targets to brain.
  Full-canvas fitted gauges describe the mathematical map; exterior coordinates
  are not observable anatomical evidence and must not receive dense tissue loss.

Reuse the three paired raw/exact-black/imperfect-brush presentation recipes of
cohort sections002, sharing anatomy, geometry, noise and pre-brush image. Apply
exact-black masks after noise; no inferred segmentation is required. Keep the
same synthetic-background limitation explicit. Save all 128 sections/384 modes,
including censored cases, in a fresh I: directory with source snapshots/hashes,
parent references, distinct new section/observation IDs and nuisance ancestry.
Do not modify the existing cohorts or any held-subject descendants.

## Intended next use and limitations

This is the smallest clean data extension for the demonstrated finite-frame
capture gap regardless of whether signed pose evidence helps. It preserves a
single coherent anatomy per subject and supplies exact targets for chart errors
that full-box training suppresses. It does not prove 128 draws are representative,
add biological independence, solve acquisition-domain appearance transfer, or
guarantee the current ribbon/updater can express and recover every target.
Retain exact slabs if a linear-director approximation or current decoder limits
prove inadequate; do not silently censor such examples or replace exact truth.

A subsequent, separately frozen learning protocol should use the same whole
coarse checkpoint and its **predicted TRAIN catalogue candidates**, retaining
failed captures; truth-near starts alone cannot establish global-to-local capture.
No evaluation generation is authorized here. Existing held-subject development
checks remain unchanged and cannot, by themselves, certify performance on this
new finite-FOV distribution. No training result, calibrated uncertainty, GUI
release, or public benchmark claim follows from preparing these views.
