# Optional joint constraints: integration gaps

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
