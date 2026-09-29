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
- Matched coordinate control002 completed all4,000 applied updates; session70095
  exited0 and the independent post-exit audit exited0. Its learning gate **fails**:
  landmark reduction6.48%, map error1.36% worse than identity, negligible normal
  improvement. Coordinate input alone provides only5.52um extra landmark gain
  over001. [Full result](LOCAL_COORDINATE_CONTROL_002_RESULT_20260929.md).
- Joint adaptation plus broad proposal rehearsal003 completed: session `38572`
  exited0 after4,000 updates,2153.03s. Launched from `8b4d58f` with
  `training/run_joint_v6_joint_rehearsal.py`. Output:
  `I:/AnatomyTracker/runs/joint_v6_joint_rehearsal_003`.
  Full640-row global and256-row local endpoints were independently audited
  after exit. Integrity and global retention pass; local/combined gates fail.
  Landmarks improve6.58%, map error worsens1.73%, global normal remains42.33deg.
  No extension/promotion. [Result](JOINT_REHEARSAL_003_RESULT_20260929.md).
- Whole parent: own-lineage `joint_v6_proposal_curriculum_003/step_20000`.
  Source/shared encoder and global proposal are now trainable along with the
  recurrent pose updater, atlas stem and SVF decoder. Each update accumulates
  local geometry loss and broad full-cell proposal NLL before one AdamW step.
  No weights from old models or independent encoder merging. This remains one
  full-model checkpoint, not continuation from the failed002 endpoint.
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
Matched coordinate control002 and whole-model joint unfreezing with global
proposal rehearsal003 are finished and audited; neither passes the local gate.
The latter ran via `training/run_joint_v6_joint_rehearsal.py`;
its [fixed protocol and retention gates](JOINT_GLOBAL_REHEARSAL_PROTOCOL_20260929.md)
start from the same whole003 parent, not the local002 endpoint. The frozen-source engineering gate must not
become a permanent prohibition on investigating feature adaptation if these
controls fail. Do not start public benchmarking or uncertainty calibration while
point learning is poor.

The proper geometric proposal-loss control 008 is implemented and documented,
but not launched or selected. An optional 896-parameter predicted-coordinate
input to the shared recurrent updater is implemented and was trained in002.
The proposed [matched local002 control](LOCAL_COORDINATE_CONTROL_002.md) is
selected after local001's failed audit and is now audited, with its failed gate
preserved. The existing independent audit script produced its completed endpoint
comparison after confirmed exit. Original local001 auditor is preserved at commit
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
adapter and first accepted subject are complete; one inspected section has27um
RMS normal residual. Sixteen-section paired-brush CPU preparation completed:
session `29655` exited0 after331.21 seconds, source `3d48a91`.
Output `I:/AnatomyTracker/data/joint_v6_coherent_subject_sections_001` is frozen.
All16 arbitrary-plane draws and48 paired observations from one shared subject
are retained, with exact3D targets and no legacy total2D-SVF labels. Five plane
draws are low-support censored in every mode;11 planes/33 observations are
support-eligible. The eligible planes' canonical normal-residual RMS ranges
22.49–45.25um. This is target geometry, not prediction error or animal validation.
The [curved-slab representation check](CURVED_SLAB_REPRESENTATION_CHECK_20260929.md)
fits a centre surface plus local through-thickness director to all16 slabs:
maximum coordinate error0.164um, on this one mild synthetic subject only.
The [physical ribbon constructor](RIBBON_ACTUAL_COORDINATE_CHECK_20260929.md)
reproduces those fitted targets to numerical precision and has a conservative
physical derivative bound. These are representation/geometry checks, not model
accuracy or biological validation. The [opt-in native recurrent ribbon path](NATIVE_CURVED_RIBBON_PATH_20260929.md)
is now implemented in the same joint model: each reflection hypothesis keeps
its own pose/field, re-renders the actual curved slab, and uses the shared updater.
It remains untrained. A bounded actual-data GPU forward/backward now passes;
a tiny FP32 normalization-order mismatch was diagnosed and corrected without
changing physical geometry or relaxing the parity gate. Existing forward and
the completed rehearsal driver are unchanged. No coherent-subject learning is
claimed yet.

The [12-subject coherent cohort](COHERENT_SUBJECT_COHORT_002_PROTOCOL_20260929.md)
is now generating on CPU from committed source `673b8fc`, session `9774`.
Output `I:/AnatomyTracker/data/joint_v6_coherent_subject_plans_002` is protected
until process exit. Eight training and four development subject maps have
preassigned identities; acceptance and completeness are not yet established.
No sections or learning are performed by this plan-generation process.

The [fresh image-key retrieval experiment](IMAGEKEY_RETRIEVAL_PROTOCOL_20260929.md)
completed4,000 updates, terminal8803 EXIT0, from `edc99b8`. Its independent
audit45886 EXIT0 passes integrity and every fixed candidate-stage gate.
Eligible group-macro normal error47.50->40.36deg and top32 physical-plane
capture10.59%->68.46% versus original003 at the matched4k budget. However,
full-frame corner capture within1mm is only4.28% at top32; this is not usable
alignment accuracy. [Full result, pins and limitations](IMAGEKEY_RETRIEVAL_001_RESULT_20260929.md).
Select its whole fresh checkpoint for conditional native ribbon learning,
without encoder merging or imported old weights. First run a truth-near
coherent-subject control and a separate raw Allen development transfer diagnostic;
neither substitutes for honest global capture or real final-test qualification.
The coherent plan-generation session9774 remains active and its tree protected.

Still required: convincing arbitrary-plane global and native joint accuracy,
usable optional constraints, genuine animal-disjoint validation, calibrated
joint uncertainty/site-region probabilities, frozen fair external comparison,
and practical integration/delivery in the existing desktop Anatomy Tracker.
