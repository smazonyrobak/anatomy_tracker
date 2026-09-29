# Coherent subjects: coordinate-rendering prerequisite — 2026-09-29

Status: one standalone numerical primitive added; existing planar renderer,
model, training driver and frozen data are unchanged. No coherent-subject model
has been trained or qualified by this change.

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
