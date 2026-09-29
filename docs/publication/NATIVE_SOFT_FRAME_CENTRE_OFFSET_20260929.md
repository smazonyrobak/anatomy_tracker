# Opt-in native soft frame-centre offset conditioning

2026-09-29. Source implementation only: not trained, not calibrated, not wired
into the inference wrapper/GUI, and not enabled by an experiment driver. No
active training outputs were accessed and no model run was launched.

## Observation and API

Construct the existing `ArbitraryPlaneJointModelV6` with
`frame_centre_offset_conditioning=True`; the default is `False`. Pass optional
`frame_centre_offset_observation` to its `forward`, to the v6 pose model's
`forward_proposal_only` / `forward_proposed`, or to native `refine` for local
learning. It is one floating tensor `[B,9]`, in this order:

```text
axis_AP, axis_DV, axis_ML,
reference_AP_um, reference_DV_um, reference_ML_um,
observed_offset_um, measurement_sigma_um, available
```

The axis is normalized internally. For physical canonical frame centre `c`,
reference `r` and unit directed axis `a`, the observed quantity is
`dot(a, c-r)` in micrometres. `c = O + U/2 + V/2` is the finite frame's physical
centre, not the tissue centroid, the mean of sampled raster pixel centres, an
electrode point, or signed distance along the candidate plane's own normal.
Changing in-plane translation can change this observation. Reversing the
declared axis requires reversing the observed offset too.

Available rows require finite coordinates, a nonzero axis, positive sigma and
availability one. Availability zero ignores all other row values, including
NaN placeholders; `None` omits the entire observation. A disabled model rejects
supplied observations rather than silently ignoring them. Sigma expresses noisy
measurement scale, not a hard interval or a calibrated model uncertainty.

## Native learned path

- `arbitrary_plane_recurrent_model_v6.py` projects six metadata features into
  the existing full-catalogue proposal context before its mixture/query heads.
  These are the unit axis, observed centre coordinate relative to catalogue
  support origin divided by 10,000 um, log(sigma / 10,000 um), and availability.
  The existing complete-catalogue normalizations and proper losses are unchanged.
- `arbitrary_plane_recurrent_model.py::refine` recomputes the current centre
  residual `(observed - predicted) / sigma` after each re-render. The six static
  features, axis coordinates in the current frame, and residual form ten inputs
  to a learned addition **before** the existing nonlinear shared GRU. Its hidden
  state drives both pose updates and the next absolute SVF decode. Final scoring
  receives the same observation. Pose/deformation gradients are not detached by
  this hook; no second network or independently deployed model is introduced.
- Joint forward compacts observations using the same ready-row indices as
  states/images. Available-row indicators are exposed in proposal/refinement
  outputs when an observation tensor is supplied.

The two new projections are zero-initialized, bias-free and created under
isolated CPU `fork_rng` scopes. Disabled construction creates no new state-dict
keys or random draws; `None` bypasses conditioning arithmetic. Existing model
weights/checkpoints retain their default architecture and initialization path.
Opting in adds only `pose_model.frame_centre_offset_proposal.weight` and
`pose_model.frame_centre_offset_evidence.weight` to a whole joint checkpoint;
continuation must explicitly initialize these new parameters while preserving
the selected complete parent model. No checkpoint loader behavior was changed.

## Probability and training boundary

This is a learned conditional distribution, not exact Bayes or repeated analytic
likelihood multiplication. Metadata can change the **complete** proposal,
including unrendered cells, once trained. The existing hybrid preserves its
selected proposal mass and unselected proposal probabilities; local refinement
still only redistributes retained mass. Nothing reclassifies the unrefined tail
as geometrically verified or calibrated. Omitted mass and raw-uncalibrated status
remain intact. No hard-bound satisfaction or contradiction detector is added.

Future training must record observed metadata separately from supervised truth,
use realistic noise scales and missingness/dropout, include wrong-but-plausible
observations, retain an observation-free control and proposal rehearsal, and
evaluate physical errors with the same information available at inference.
Exact truth offsets are not inference metadata. The frozen local pack contains
no independent real offset measurements. No surgical ray/angle, electrode-mark,
slice-order, shared-normal, GUI coordinate conversion or hard-mask integration
is implemented by this change. The source was reviewed without numerical runs;
runtime behavior and learning benefit remain to be checked in a bounded follow-up.
