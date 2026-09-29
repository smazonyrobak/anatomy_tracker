# Current joint-model development — 2026-09-29

The native goal is active. No heartbeat automation is used. This is an unfinished
model-development goal, not a shipped or benchmark-qualified system. All work,
data, temporary files and runs belong on `I:`.

## Current experiment decision

- Local001 (`I:/AnatomyTracker/runs/joint_v6_local_refinement_001`) completed
  all 4,000 applied updates; session `6220` exited zero. The independent audit
  exited zero and authenticated a **failed** local-learning gate: landmarks
  improve only 6.07%, map error worsens 4.08%, normals barely improve. Large
  CCF gains primarily resolve reflections; unreflected rows regress overall.
- [Full result and receipts](LOCAL_JOINT_REFINEMENT_AUDIT_20260929.md).
- Matched coordinate control002 is LIVE: terminal session `70095`, Python PID
  `20152`, launched 2026-09-29 13:04:39 Warsaw from commit `866ee1e` via
  `training/run_joint_v6_local_refinement.py`. Output:
  `I:/AnatomyTracker/runs/joint_v6_local_coordinate_control_002`.
  Initial full development evaluation and 100 applied updates are confirmed on
  stdout. Recheck the same handle; do not restart on an observation timeout.
  Never access this output tree while the process is active.
- Whole parent: own-lineage `joint_v6_proposal_curriculum_003/step_20000`.
  Global proposal/source encoder frozen; native shared recurrent pose updater,
  atlas stem and SVF decoder learn together. No weights from old models or
  independent encoder merging. This is one full-model checkpoint.
- Explicit limits: truth-near starts, known synthetic PSF, one atlas with
  organizational group IDs, uncalibrated probabilities. This does not measure
  image-selected global capture or biological-animal generalization.

Protocol, paired geometric baselines and fixed continuation criteria:
[local joint stage](LOCAL_JOINT_REFINEMENT_STAGE.md).

## Completed comparison decisions

- The original 43.7265-degree normal error was independently reproduced. No
  simple degrees/radians, normal-sign, axis or row-pairing bug was found.
- Both FP32 and AMP fresh eight-case controls memorize all eight full cells;
  broader orientation learning remains inadequate, not explained away by that.
- 005 normal-loss weighting gives a modest final angle improvement but worsens
  joint NLL, offset and imperfect-brush capture. 003 remains the combined base.
- 006 direct normal readout and 007 residual spatial depth both failed matched
  4k extension criteria. Neither is extended or selected. The 007 independent
  audit is committed as `9c366b9`; poor capture also persists in support-eligible
  pose and joint subsets. Architecture changes are experiments, not assumed gains.

## Next decisions, not completion claims

The local001 audit is complete; its weak geometry prevents qualification.
Matched coordinate control002 is running. Whole-model joint unfreezing with global
proposal rehearsal is prepared in `training/run_joint_v6_joint_rehearsal.py`;
its [fixed protocol and retention gates](JOINT_GLOBAL_REHEARSAL_PROTOCOL_20260929.md)
start from the same whole003 parent, not the local002 endpoint. The frozen-source engineering gate must not
become a permanent prohibition on investigating feature adaptation if these
controls fail. Do not start public benchmarking or uncertainty calibration while
point learning is poor.

The proper geometric proposal-loss control 008 is implemented and documented,
but not launched or selected. An optional 896-parameter predicted-coordinate
input to the shared recurrent updater is implemented and is being trained in002.
The proposed [matched local002 control](LOCAL_COORDINATE_CONTROL_002.md) is
selected after local001's failed audit and now running. The existing independent
audit script is prepared for its completed endpoint comparison but must not be
executed before confirmed exit. Original local001 auditor is preserved at commit
`4c3a6f0`.

Soft frame-centre offset conditioning now has an opt-in native proposal/GRU
path, disabled and untrained. It is not GUI capability or calibrated inference;
hard slice bounds, surgical ray/entry/angle and coherent stack inference still
need their correct native wiring. Exact limitations and conditioned-tail issues
remain documented in [constraint gaps](JOINT_CONSTRAINT_INTEGRATION_GAPS_20260929.md).
Use `final_pullback_map_yx_px` for observed mark → atlas-raster mapping, then
reflection → physical OUV; the current forward-map field is only an alias.

Coherent synthetic subjects require a representation extension: nonlinear3D
anatomy maps a cut plane into a curved CCF surface, not just an in-plane2D warp.
The [coordinate-grid PSF rendering prerequisite](COHERENT_SUBJECT_COORDINATE_RENDERING_20260929.md)
and [conservative full-plane sampler](COHERENT_SUBJECT_PLANE_SAMPLING_20260929.md)
are implemented separately from active training. The exact subject-coordinate
adapter is in progress. No curved-surface predictor or coherent-subject learning
is claimed from these primitives.

Still required: convincing arbitrary-plane global and native joint accuracy,
usable optional constraints, genuine animal-disjoint validation, calibrated
joint uncertainty/site-region probabilities, frozen fair external comparison,
and practical integration/delivery in the existing desktop Anatomy Tracker.
