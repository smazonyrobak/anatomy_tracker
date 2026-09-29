# Current joint-model development — 2026-09-29

The native goal is active. No heartbeat automation is used. This is an unfinished
model-development goal, not a shipped or benchmark-qualified system. All work,
data, temporary files and runs belong on `I:`.

## Live experiment

- Driver: `training/run_joint_v6_local_refinement.py`.
- Output: `I:/AnatomyTracker/runs/joint_v6_local_refinement_001`.
- Unified terminal session `6220`, Python PID `4104`, launched from source
  commit `338c636`. Confirm live/terminal state before taking further action;
  these identifiers are a handoff, not proof of continued execution.
- Full 256-section initial evaluation and 100 applied updates were observed on
  stdout. The fixed endpoint is 4,000 steps. Read stdout/process/GPU telemetry
  while active; do not inspect or modify its output tree until confirmed exit.
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

After local training exits, audit raw paired pose/deformation/CCF errors,
reflection and brush subsets, topology and frozen-parent equality. If local
learning meets its criteria, proceed to whole-model joint unfreezing with global
proposal rehearsal; otherwise address the observed failure first. Do not start
public benchmarking or uncertainty calibration while point learning is poor.

The proper geometric proposal loss is a documented, **unimplemented** possible
control, not a queued GPU run. Constraints are currently standalone factors,
not integrated v6 capability; their exact missing wiring and conditioned-tail
limits are documented in [constraint gaps](JOINT_CONSTRAINT_INTEGRATION_GAPS_20260929.md).
Use `final_pullback_map_yx_px` for observed mark → atlas-raster mapping, then
reflection → physical OUV; the current forward-map field is only an alias.

Still required: convincing arbitrary-plane global and native joint accuracy,
usable optional constraints, genuine animal-disjoint validation, calibrated
joint uncertainty/site-region probabilities, frozen fair external comparison,
and practical integration/delivery in the existing desktop Anatomy Tracker.
