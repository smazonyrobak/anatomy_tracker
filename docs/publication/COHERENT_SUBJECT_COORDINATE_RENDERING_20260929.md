# Coherent subjects: coordinate-rendering prerequisite — 2026-09-29

Status: standalone numerical renderer and offline subject-coordinate adapter
added; existing planar renderer, model, training driver and previous frozen data
are unchanged. No coherent-subject model has been trained or qualified by this
change.

## Source-derived gap and reuse

The current finite joint generator assigns animal groups using
`sample_index // sections_per_animal`, but draws each row's rendering and
synthetic-processing seeds separately (`arbitrary_plane_finite_joint_curriculum_v5.py`,
`_make_parent_and_slab` and `make_finite_joint_curriculum_training_rows_v5`). Those
groups do not constitute distinct shared 3D anatomies.

Existing reusable implementations already provide the missing anatomy source:

- `sample_animal_subject_deformation_plan_v2`: one animal-index-seeded positive
  global scale and affine-free coarse/fine 3D B-spline stationary velocity field,
  with exact accepted coefficients, integration settings and plan/realization IDs.
- `subject_to_ccf_points_v2` and `make_subject_slab_render_v2`: map all physical
  subject-space PSF queries through the same frozen animal map before atlas sampling.
- `fit_subject_centre_plane_and_residual_v2`: uniform full-canvas OUV fit plus the
  complete 3D residual field, including normal and tangential residual diagnostics.

A nonlinear 3D subject map generally turns a physical cut plane into a curved
CCF surface. Current OUV plus in-plane 2D SVF cannot represent its normal residual;
mapped thickness queries also need not follow the fitted CCF plane's normal.
The older 2D section-processing SVF is not the missing 3D anatomy deformation.
Relabelling the fitted plane and that SVF as exact complete truth would therefore
be incorrect. Histological slice-to-volume work explicitly evaluates 3D spline
warps; this supports considering a 3D surface mapping, not a performance promise
for this model ([Osechinskiy and Kruggel, 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3335496/)).

## New primitive and exact coordinate convention

`render_finite_thickness_coordinate_grid(volume, coordinates, origin, spacing, weights)`
in `training/arbitrary_plane_full_frame_primitives.py` takes:

- Continuous scalar/feature volume `[C,AP,DV,ML]`.
- Already-composed physical query points `[B,S,H,W,3]`, AP/DV/ML in micrometres.
- Physical voxel-face origin and spacing; voxel centre `i` is
  `origin + (i + 0.5)*spacing`.
- Positive PSF weights `[S]` or `[B,S]`, normalized globally over S for each item.

It returns `[B,C,H,W]`, using the same trilinear `grid_sample`,
`align_corners=True`, and zero padding as the unchanged planar renderer. There
is no tissue/in-bounds renormalization, inferred normal, reflection, or hidden
deformation. Non-floating semantic labels still require a separate nearest-label
sampling path; this primitive is not a label renderer. Valid shapes and positive
weight mass are caller preconditions, not a new validation framework.

For existing **observed-raster, reflection-conjugated** section pullbacks the
correct upstream composition is:

`observed x -> pullback_observed(x) -> finite raster reflection -> subject plane + z*subject_normal -> subject_to_CCF`.

Finite reflection means `x -> W-1-x`, not `W-x`. Reflection can precede the
processing map only when that map is explicitly the canonical, pre-conjugation
map. Do not conjugate existing observed-raster truth a second time. The physical
PSF offset is inserted in subject space before the 3D map. The new primitive
only samples supplied coordinates and imposes none of these upstream choices.

## Bounded CPU comparison on completed actual inputs

The single check decoded the complete pinned Allen atlas through existing v6
preprocessing, obtaining `[2,528,320,456]` float32 intensity/support channels,
array receipt SHA-256
`255c2df028b8e973305d5edb2eb26dadd8c96530c638a28aec5183be83ffd832`.
It used rows **1 and 2** from the completed
`I:/AnatomyTracker/data/joint_v6_local_refinement_frozen_001/training.pt`, with
their actual full-frame states and nine-sample PSF offsets/weights. The receipt
retains exact row identities, states and PSFs. Query shape was `[2,9,96,96,3]`.

- Planar outputs were **bitwise equal**, maximum absolute difference **0**.
- Shared `[S]` weights and per-row `[B,S]` weights both matched the old renderer;
  multiplying all weights by 16 preserved the output exactly.
- **52,178** queries fully beyond the trilinear volume footprint sampled exact
  zero before PSF reduction.
- Coordinate, state and weight gradients were finite/nonzero: L1 norms
  **0.0006478814**, **0.8123623**, and **0.008862287**, respectively.

No GPU, optimizer, active-run output, or fabricated fixture was used. This
checks the planar special case and differentiation, not a nonlinear subject
generator, a curved-surface model, PSF quadrature convergence, or learned accuracy.
Volume gradients were not separately exercised in this bounded check.

[Check script](I:/AnatomyTracker/tmp/check_finite_coordinate_grid_20260929.py),
SHA-256 `7c4dbc5535d0010a7ff73a7a8020fe95f01d7f0d241fd8cc30a554987620ad86`.
[Result receipt](I:/AnatomyTracker/tmp/check_finite_coordinate_grid_20260929.json),
SHA-256 `b01e883f62bcc7646be1a4984674b1abe8e614db60e79c60291d90193edb68da`.
The receipt records the primitive-source hash.

## Offline subject-coordinate adapter

`training/arbitrary_plane_coherent_subject_v6.py` adds one function:

```python
make_coherent_subject_section_v6(
    subject_plan, subject_ouv_ap_dv_ml_um, observed_pullback_yx_px,
    reflection_xy, axial_offsets_um, axial_weights,
    section_identifiers, source_identifiers, volume_c_ap_dv_ml,
    origin_ap_dv_ml_um, voxel_size_ap_dv_ml_um, *, mapping_batch_size=8192,
)
```

The caller supplies one accepted, frozen v2 plan, authenticating persisted plans
at the load boundary. A freshly accepted sampler result needs no redundant
per-section plan replay. The existing exact NumPy/RK4 numerical mapper is reused;
this is offline target generation, **not an autograd subject-flow implementation**.
The function chooses no planes, support bounds, acquisition distribution or
deformation seeds. It requires the section's animal/synthetic-animal/split
identifiers to match the supplied plan and retains separate source ancestry.

Physical OUV is `[3,3]` or flattened9, absolute observed pullback is y/x `[2,H,W]`,
and `reflection_xy` contains horizontal/vertical Boolean flags. Offsets and
positive weights are `[S]`, including an exact zero centre sample. Voxel and
raster conventions are those above. The plan/realization/context/seed reference,
input OUV, original 2D processing map, reflection and PSF are retained.

Crucially, two distinct gauges are exported:

- `canonical_anatomy_centre_ccf_ap_dv_ml_um_float64` and
  `canonical_anatomy_plane_fit` map the original canonical subject plane without
  section processing or reflection. The latter contains its full-canvas fitted
  OUV and complete anatomy-only 3D residual.
- `target_centre_ccf_coordinates_ap_dv_ml_um_float64`,
  `target_psf_ccf_coordinates_ap_dv_ml_um_float64`, and
  `observed_total_map_plane_fit` describe the processed/reflected observation.
  The latter fit is a separate total-map diagnostic, not the canonical pose
  target. Neither residual is a 2D stationary velocity.

`raw_rendered_channels` samples the continuous supplied coordinates, not a
bilinear warp of a previously rasterized image. The original processing map and
its in-raster domain mask are separate outputs. No implicit finite-FOV mask,
tissue mask, damage, crop, appearance synthesis or smart-brush operation is
applied. Those observation operations and their validity weights remain later
stages; hard semantic labels still need nearest-label sampling.

The first real Allen-domain plan uses root seed `2026092907`, training animal
index `0`, and animal ID `joint-v6-coherent-pilot-001-animal-00000000`. Its
generation/check script is
`I:/AnatomyTracker/tmp/check_coherent_subject_adapter_20260929.py`; it persists
under `I:/AnatomyTracker/data/joint_v6_coherent_subject_pilot_001` using the existing
v2 bundle representation: `subject_plan.metadata.json` with nested array
references, `subject_plan.arrays.npz` with all numerical arrays, and
`subject_plan_receipt.json`. The existing `_read_raw_artifact` restores that
representation with `allow_pickle=False`. Reuse this exact plan across sections;
do not regenerate a new plan from a section seed. Persistence alone is not an
independent plan replay or a learned-model qualification.

### First actual subject and section composition

The CPU process exited **0**. The existing default v2 sampler accepted amplitude
**62.5 um**, taking **264.7 s** (complete context/plan/section check **296.7 s**).
The pinned raw Allen scalar volume was `[528,320,456]` at 25 um spacing; prepared
v2 context SHA-256 is
`c3bd31cc81af2788437cfd064f4cbf44d1d9f111919031c4c691552e796d94a8`.
Plan ID is `379a1cddefaede38495ee0910e911426159551d6f4b00adad55cd8f1b1e0dc18`;
plan receipt SHA-256 is
`496669aacf08f9a5cee3aedc4fcf3a600cfe9ced106fd1358c3d4bef2cf8c7bc`.
Its complete accepted arrays and source snapshot are retained, not only its seed.

Completed frozen local-training **row 3** supplied its exact OUV, nine-sample PSF,
observed-conjugated pullback and horizontal reflection. These are processing and
geometry ancestry only: the image and 3D anatomy targets were newly rendered for
the new subject, with distinct animal/specimen/experiment/section IDs. No old
tissue validity, segmentation or deformation-eligibility label was copied.

- Canonical anatomy-only residual RMS **50.7126 um**, including normal RMS
  **27.0245 um** and maximum absolute normal residual **93.0627 um**.
- Observed total-map residual RMS **86.0717 um**; its normal RMS is **26.8788 um**.
  These different gauges are retained separately, not conflated as one SVF.
- Canonical and observed `fitted_grid + residual` reconstruct their respective
  exact grids to **3.55e-15** and **1.42e-14 um** maximum absolute error.
- The independently evaluated centre mapper point agrees exactly; the original
  processing map is bitwise preserved, and finite reflection-after-map order was
  checked against the exact subject-space centre coordinates.
- Exact PSF target shape is `[9,96,96,3]`; the finite raw render is `[1,96,96]`,
  containing **5,237** nonzero pixels. This count is not a tissue eligibility test.

This single composition demonstrates the nonplanar target gap; it is not a
cohort, sampler-coverage assessment, validity certification for reused processing
maps on new anatomy, biological validation, or trained-model result. Subsequent
coherent rows must derive their observation masks and validity from their newly
mapped anatomy. No independent plan replay or PSF-convergence check was added.

[Frozen result](I:/AnatomyTracker/data/joint_v6_coherent_subject_pilot_001/check_result.json),
SHA-256 `cce2057428b2caed178a6320cd9f79f9bcaab2003308dfcb6630b003e73644b9`.
It records exact source and artifact hashes; check-script SHA-256 is
`16dd3bba49d920d1c556b96cfd2b21b4a3e689ecd65d7102f503d045f30c8f0d`,
adapter-source SHA-256
`b0b9601d03eb3ea4830fb989d8a5e0f68f3b85379464ad9dcfba8c7c6caa5333`.

## Remaining bounded integration work

Freeze one subject plan per animal across all its sections; retain independent
section/acquisition seeds and separate 2D section-processing truth. Export exact
centre and PSF CCF grids, fitted-frame gauge, complete 3D residual, validity/weights,
atlas identity, animal/specimen/experiment/section IDs, subject plan/realization
IDs and coefficient hashes. Keep all descendants of one subject in one split.

Use subject-aware conservative support bounds, including scale/displacement
bounds, to retain coverage of every brain-intersecting subject plane. Do not
reuse the old `subject_support_resolution_v2` pose-redraw wrapper unchanged:
it samples original-atlas support and retries full poses. Preserve reference
RP2 orientation/roll coverage and disclose conservative-bound offset sampling
and censored/marginal rows. These are synthetic subjects from one source atlas,
not independent biological animals or evidence of biological generalization.
