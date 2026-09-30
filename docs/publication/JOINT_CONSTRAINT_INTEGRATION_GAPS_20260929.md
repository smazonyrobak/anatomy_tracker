# Optional joint constraints: integration gaps

## 2026-09-30 native-v7 update

Native curved point mapping and an experimental GUI inference/save/reload path
now exist and passed an offscreen execution check. Constraints remain untrained
and explicitly unused; the v6-specific observations below are historical.

The GUI supplies elevation bounds, not a measured3D direction/azimuth. Before
constraint training, redefine the currently unused12-value context as AP
bounds/availability3, entryAPML/radius/availability4, elevation bounds/availability3,
and maximum-depth/availability2. Keep tip-depth/end-point knowledge separate.
The legacy AP bound is a coronal reference-plane parameter, not arbitrary-image
centreAP: define the intended anatomical anchor explicitly before using it.

Stereotaxic-to-physical CCF conversion is `25*(bregma_voxel+.5)` plus
`diag(-1,-1,+1)*stereotaxic_um`; directions omit translation. Preserve raw marks
and their probe IDs through the native curved map, without a second reflection.
The12-value context alone does not establish coherent multi-probe conditioning.

Use one independently sampled subject-space trajectory per subject/probe and
derive observed marks from its finite-section intersections, crop/damage/brush,
and separately sampled observation noise. A straight subject-space probe can
map to a curved CCF path: do not penalize this correct deformation with a hard
straight-CCF-ray factor. Model shared surgery once per probe; independently
numbered mixture modes across slices are not coherent joint animal hypotheses.
The GUI still needs explicit animal/specimen/experiment IDs for such grouping.
Train missing and conflicting constraints, retain unconstrained inference, and
never reinterpret a hard tolerance as a Gaussian standard deviation.

## Original v6 review

2026-09-29. Read-only source review; no trained constraint capability or
calibrated uncertainty is established. No active global/local training outputs
were accessed for this review.

## Existing code and missing wiring

`training/arbitrary_plane_joint_constraints.py` supplies differentiable
`electrode_ray_factor`, `entry_position_factor`, `attack_angle_factor`,
`hard_constraint_factors`, and mark-to-CCF mapping. The factors distinguish a
finite physical ray, Gaussian measurement uncertainty and exact hard bounds.
`training/arbitrary_plane_constraint_posterior.py::constraint_conditioned_posterior`
combines explicit coherent section/ray hypotheses, preserves reflected modes,
counts shared surgery once per animal and distinct marks once each, returns
residuals, and flags all-infeasible supplied hypotheses.

These primitives have no callers in v6 model/inference/training/loss or the GUI.
`ArbitraryPlaneJointModelV6.forward`, `ArbitraryPlaneRecurrentModel.refine`, and
`run_arbitrary_plane_inference_v6` accept no optional constraint observations.
There is no v6 partial-stack-order or shared-antipodal-normal factor. The GUI
still uses the legacy coronal `solve_ordered_lattice`,
`solve_probe_constrained_lattice` and `refine_pose_search` in
`source/proprietary_trajectory_tool.py`.

The current frozen training pack contains no real coherent stack/ray observations
or electrode marks. Organizational synthetic group IDs do not create those
observations. Future simulated constraints must be noisy/missing observations
from an explicit generative process, not exact pose/trajectory labels supplied
as inference inputs. Surgical attack angle constrains the directed **probe ray**,
never the antipodal slice normal. Entry/angle metadata alone cannot locate an
unobserved electrode in an image.

## Correct map adapter

The following pullback/OUV adapter applies to the **legacy planar/SVF path only**.
The later native curved-ribbon path needs the observed centre-surface adapter
described below; reducing its geometry to a flat OUV would discard deformation.

Use `output["refined_output"]["final_pullback_map_yx_px"]`: querying an observed
source pixel returns its fixed atlas-raster coordinates. Apply the selected
representation affine **after** this map, then the canonical physical frame with
`x/W,y/H`. Use the reflection-aware posterior primitive, not the simpler
`marked_points_to_ccf` without explicitly accounting for representation.

This direction was traced through implementation, not inferred from names:

- `AffineFreeSVFDecoder.forward` in
  `training/arbitrary_plane_deformation_primitives.py` computes
  `(forward, inverse) = (exp(v), exp(-v))` and assigns both
  `forward_map_yx_px` and `pullback_map_yx_px` to the same `forward` tensor.
- `training/arbitrary_plane_finite_joint_curriculum_v5.py` negates the generator
  velocity before certification against `source_to_fixed_map`; the decoder's
  learned `v` therefore already has source-to-fixed pullback direction.
- `ArbitraryPlaneJointModelV6._legacy_joint_output` preserves the aliases by
  taking the final element of each sequence. Thus `final_forward_map_yx_px`
  currently equals `final_pullback_map_yx_px`, but the latter is clearer.

Do not use `final_inverse_map_yx_px`, negate velocity again, or conjugate the
already reflected source SVF twice. At inference the dense-truth feedback gate
must be absent; truth censoring is not an observable input.

## Retained hypotheses are not the full constrained posterior

For honest, non-teacher-forced v6 inference, the existing refined joint weights
are conditional within retained top-K. Their full-proposal weights are

```python
log_weight = (
    output["refinement_retained_probability"].log()[:, None, None]
    + output["refined_output"]["pose"][
        "conditional_within_topk_representation_log_probability"
    ]
)
```

Combine these with `final_cell_state`, the final pullback and selected catalogue
representation affines. A normalized retained table can enter
`constraint_conditioned_posterior` only with an explicit **conditional on
retention** scope. Omitted cells have no refined warp or ray likelihood. After
conditioning, their original omitted mass is neither known to remain unchanged
nor justified to become zero. Report pre-constraint omitted mass and
`conditioned tail unresolved`; do not claim a full posterior or region confidence.
Global conditioning needs tail-likelihood evaluation or defensible bounds and
candidate expansion. All retained modes failing means no feasible retained
hypothesis, not proof that the full continuous problem is contradictory.

For multiple sections, construct explicit joint tuples of section cell/reflection
choices and a shared ray latent, retaining joint selection mass. Unrelated top-K
columns are not coherent hypotheses. Do not multiply independently top-K-normalized
section posteriors and treat the result as the animal posterior. Ray uncertainty
must be inferred/marginalized, not silently fixed from unavailable truth. Recompute
constraint factors from the original image prior after updates; feeding the
conditioned posterior back as that prior would count evidence repeatedly.

## Minimum next change after useful local learning

Start with optional **single-section physical-offset conditioning**: declared
axis/reference, measured offset and explicit measurement sigma, plus a separate
optional hard interval. This requires no invented track or coherent stack.
Keep one whole-model lineage. Add standardized residual/availability conditioning
to existing recurrent evidence, and apply/recompute the same physical factors
before candidate pruning and after pose/warp updates. Train paired absent,
realistically noisy and incorrect observations with metadata dropout and proposal
rehearsal; absent observations must preserve the original path. Supervised truth
remains a target, not an exact constraint input.

Two concrete prerequisites prevent dishonest shortcuts:

- `hybrid_full_catalogue_posterior_v6` currently rejects any `-inf` proposal.
  Make feasible masking and all-infeasible handling explicit before adding hard
  exclusions; do not replace exclusions with a merely large finite penalty.
  A narrow interval missing coarse cell centres is not by itself continuous
  infeasibility: preserve intersecting cell support or expand/refine candidates.
- Preserve unresolved-tail accounting and recheck hard feasibility after local
  updates. Learned penalties alone do not guarantee bounds; abstain when feasible
  candidates cannot be established, rather than silently relaxing constraints.

Subsequent marked-track conditioning can reuse the physical residual primitives
inside the same recurrent update once explicit shared-ray hypotheses and valid
observations exist. The new factors use physical CCF AP/DV/ML (+DV ventral),
whereas `source/probe_constraints.py` reverses stereotaxic AP/DV signs relative
to atlas coordinates. The GUI adapter must convert coordinates, directions and
covariances consistently with the atlas origin/voxel-centre convention. GUI entry
radius and angle tolerance remain hard bounds, not implicit Gaussian sigmas;
cortical-surface entry and depth validity also require explicit ray hypotheses.

This review authorizes no automatic constraint correction, probability claim,
benchmark or deployment. Constraint calibration and biological-animal validation
remain separate requirements.

## GUI audit after the curved-ribbon extension

Read-only inspection of `source/proprietary_trajectory_tool.py` and
`source/probe_constraints.py` establishes these existing inputs:

- Per-probe surgical AP/ML, entry-disk radius, attack angle/tolerance and optional
  maximum depth. Radius, tolerance and maximum depth are **hard bounds**, not
  Gaussian standard deviations. Entry DV comes from the dorsal Isocortex surface.
- Attack angle is0deg horizontal/90deg vertical. No measured azimuth or physical
  multi-shank roll is supplied. The displayed fitted horizontal bearing is not
  a shank-orientation measurement.
- Endpoint choices distinguish known surface-to-tip depth from deepest marked
  dot as tip. Neither is interchangeable with maximum allowed insertion depth.
- Marks in `ProbeTrace.slice_points` are downsampled, unrotated `raw_display`
  xy pixels. Brightness weights are not calibrated localization uncertainty;
  derived `atlas_points`/`volume_points` are not extra independent observations.
- Batch AP bounds and checked anterior-to-posterior ordering exist, but not
  cutting thickness, spacing or arbitrary stack-normal measurements. Legacy AP
  bounds refer to plane AP at atlas-centre DV/ML, not finite-frame centre AP or
  arbitrary-plane normal offset. Native controls need an explicitly labelled,
  versioned anchor rather than silent reinterpretation.

`session_points_to_volume` currently evaluates only flat
`O+(x/W)U+(y/H)V`, then divides physical CCF micrometres by25 and subtracts0.5.
Native marks must instead use recorded raw-display-to-model-raster coordinates
and interpolate the **final observed centre CCF surface**. Reflection is already
spatially applied in that observed surface: do not flip again or pass through
the old2D warp plus flat plane. Canonical OUV remains a pose reference/legacy
fallback, not the complete anatomical map.

The verified stereotaxic conversion, for voxel-face physical origin0, is
`p_CCF_um = 25*(bregma_voxel + 0.5) + S*p_stereotaxic_um`,
`S=diag(-1,-1,+1)`. Directions use`S*d`, covariances`S*cov*S.T`.
Bind the exact atlas/bregma convention and input-raster transform to predictions.
The session archive does not yet expose explicit biological animal/specimen/
experiment IDs; obtain those from actual metadata, not filename inventions.

The delivery contract requires one shared directed ray and endpoint latent per
animal/probe, coupled to section surface hypotheses through original marks.
Count shared surgery once per probe and each independent mark once. Propagate
joint section/ray/endpoint samples to sites without converting hard tolerances
into uncertainty. The legacy coronal/outline-entry gate must not make brush
segmentation mandatory for native inference. None of this GUI wiring is claimed
implemented or calibrated by this source audit.

### Native forward mark adapter implemented separately

`training/arbitrary_plane_ribbon_marks.py::raw_marks_to_ribbon_ccf` now supplies
the forward map from original raw-raster marks through an explicit recorded
raw-to-model affine to each retained observed curved CCF surface. It preserves
every cell/reflection hypothesis; it never averages their coordinates. Raw and
model pixel-centre bounds, availability and finite coordinates determine validity.
Invalid points return NaN, including the half-pixel fringe outside the tabulated
surface-centre hull. No segmentation is required and no second reflection,
legacy warp or flat OUV lift is applied.

A single CPU numerical check used the completed TRAIN observation592 prediction
`joint_v6_imagekey_ribbon_inference_preflight_001/prediction_chunk_1.pt`, SHA256
`ac60e693d2efc496efb3d1a286c0f135d0807806e38eeebbcf8f012687b1d747`.
Across all4 cells and2 reflections, 9,216 integer centres and9,025 half-centres
per hypothesis agreed with direct indexing/four-corner interpolation: maximum
FP64 errors4.184e-11um/1.819e-12um; FP32 integer error.000977um. Seven
out-of-hull, nonfinite or unavailable queries were invalid/NaN. Adapter source
SHA256 is `b4c7b39fdd146a92b1ed631b33ac3c92d6987e52ee02f385cb6668d1dc5531ea`.
This checks interpolation on actual saved surfaces, not electrode annotations,
registration accuracy or calibrated uncertainty.

GUI wiring remains absent. Persist the exact `raw_display`-to-native-input affine
with preprocessing provenance; `display_scale` alone is insufficient. The adapter
does not provide atlas-click-to-raw inversion. A future native GUI branch must
not silently reuse the legacy inverse, or the existing early return that requires
a legacy transform before probe points can be recomputed.

### Exact GUI input-affine contract: source findings, not implemented

There is currently **no GUI native96x96 preparation/inference branch**.
In `source/proprietary_trajectory_tool.py`, `_load_session_image` (6377)
downsamples the source, then makes percentile-normalized grayscale
`raw_display` (6390–6394). Probe marks remain in this raster;
`_slice_raw_to_display_points`/`_slice_display_to_raw_point` (7127/7130)
apply/invert `session.slice_transform`. `slice_geometry_matrix` (1156)
constructs rotation about pixel centres, bounding-canvas translation and flips;
`transform_slice_image` (1186) returns that same raw-to-oriented matrix.

The smallest future attachment is a native-preparation sibling beside
`prepare_pose_inputs` (419), using the orientation/explicit-mask snapshot at
`_automatic_alignment_image_job` (8710). Capture the matrix from the actual
image-transform call rather than discarding it as the legacy call does (453).
Do not route native inputs unchanged through the legacy preparation: its
`registration_brain_mask` (363) falls back to automatic segmentation, and
its crop (465–477) depends on that mask. With no explicit mask/crop, retain
the full oriented image and zero outline/availability; do not invent a mask.

For oriented crop bounds `(x0,y0,x1,y1)`, width `w=x1-x0`, height `h=y1-y0`,
and a96x96 half-pixel-centred resize, record the column-vector affine:

```text
T = raw_display -> oriented_display, from the actual image transform
C = [[1,0,-x0], [0,1,-y0], [0,0,1]]
sx = 96/w; sy = 96/h
D = [[sx,0,(sx-1)/2], [0,sy,(sy-1)/2], [0,0,1]]
raw_to_model_xy = D @ C @ T
```

Do not multiply by `display_scale`: marks are already downsampled, and that
nominal factor alone also omits resize rounding. The recorded crop/resize must
be the one that produced the input, not a later reconstruction from GUI state.
Smart-brush selection masks are already **oriented-display** arrays, produced
from `session.rotated` in `_compute_smart_surface_selection` (7513) and
transformed old-to-new when geometry changes (6771–6781). Crop/resize them once;
do not reapply `T`. Use centre-consistent nearest-exact mask sampling on the same
resize grid, derive the binary boundary at96x96, and impose exact black exteriors
after image resampling. Raw outline points/strokes, unlike the mask, require `T`.

Persist the matrix with raw-display/oriented/model shapes, actual crop, input
mode, explicit-mask and image hashes in each native-input record. The existing
`auto_alignment_diagnostics["native_input"]` can retain that record without a
new session field: diagnostics are already in `SESSION_STATE_FIELDS` (135),
saved at5646 and restored at5711. Existing full-plane geometry/mask invalidation
(6796/7222) must also invalidate the corresponding prediction/input binding.
Pass the recorded matrix to `raw_marks_to_ribbon_ccf`; its observed surfaces
already include reflection and must not be flipped again.

**Photometry remains unresolved.** `_update_slice_image` (6854) makes
`weight_image` from normalized `raw_display`, whereas `session.rotated` also
contains the user's brightness LUT. Neither is equivalent to the current real
Allen training contract: red/255, acquisition-centred physical sampling,
antialiasing and border-median padding (`prepare_joint_v6_allen_training_inputs.py`
58,145–146). Avoid accidentally feeding display-LUT pixels, but do not call
`weight_image` training-matched. This note fixes coordinate bookkeeping only;
it neither selects a validated GUI photometric policy nor implements native
inference, constraints, confidence calibration or deployment.
