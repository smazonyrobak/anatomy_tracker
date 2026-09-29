# Current joint-model development — 2026-09-29

The native goal is active. No heartbeat automation is used. This is an unfinished
model-development goal, not a shipped or benchmark-qualified system. All work,
data, temporary files and runs belong on `I:`.

## Latest handoff

- Image-key001 is completed and independently audited: candidate-stage gate
  passes, but MAP orientation40.36deg and full-frame1mm top32 capture4.28%
  remain inadequate. Its whole step4,000 checkpoint is the native ribbon parent.
- The raw Allen diagnostic, terminal52827 EXIT0 from `c9a94c6`, completed64
  images from6 development donors. Donor-macro normal error45.05058deg,
  normal-offset error4150.58um, top32/top128 plane capture9.09091%/21.36364%.
  Synthetic improvement has not established raw-image transfer. These upstream
  affine references are not expert anatomical ground truth or a final benchmark.
- Coherent-plan generation9774 and section generation68817 both exited0.
  The frozen cohort retains640 sections/1,920 observations:512 train sections
  from8 synthetic subjects and128 development sections from4 separate subjects,
  with zero rejected draws. Section generation took1343.483s.
- Target-capacity audit36825 exited0 and authenticated all640 section artifacts.
  Surface/director maxima150.933um/.148847 and derivative bound.178064 remain
  below native200um/.2/.35 limits; no cap or rescaling activates. Maximum
  through-slab linear-fit point error is.257013um. This is target geometry,
  not decoder learnability or biological validation. [Capacity result](COHERENT_RIBBON_TARGET_CAPACITY_002_RESULT_20260929.md).
- Optional-outline-dropout comparison65125 and its independent audit completed.
  Integrity passes but the dropout improvement gate fails; do not promote B or
  tune on the six development donors. Whole control A6k is the next real-training
  warm start, not a qualified model. [Audited result](IMAGEKEY_OUTLINE_DROPOUT_001_RESULT_20260929.md).
- Real+synthetic001 is **active**, session42210 from `3d37b71`, continuing whole
  A6k. It pairs256 unchanged real training images/58 donors with continuous
  upstream affine atlas renders and arbitrary-plane synthetic replay. No dense
  real warp labels or registration covariance are fabricated.
- Native conditional pose+ribbon001 is **active**, session89162 from `dc32605`,
  with the completed cohort bound and the original whole image-key4k parent.
  It is a separate truth-near, known-PSF local control, not global inference.
  No independently trained encoders are merged.
- Both active training output trees and operative sources are protected until
  their respective exits; inspect only live stdout/telemetry. Independent endpoint
  audits follow exit. Concurrent timings are not hardware benchmarks.
- After real+synthetic exits, independently audit its endpoint and apply the
  frozen gates. After native exits, audit the raw held-out curved coordinates
  and geometry, then decide actual retrieved-beam initialization versus revised
  learning. A successful real warm start needs subsequent native training of
  that **whole model**, not splicing in this independently trained native control.
- No heartbeat or additional user prompt is needed for these native-goal steps.

## Completed planar/SVF experiment decisions

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
  Source/shared encoder and global proposal were trainable along with the
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
For native ribbon inference, map raw-display marks through the recorded
raw-to-model raster transform, then interpolate the **final observed centre CCF
surface**. Its raster reflection is already applied: do not flip again or reduce
the surface to a flat OUV. The pullback→reflection→OUV adapter applies only to
the legacy planar/SVF path. Neither adapter is yet a calibrated GUI integration.

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
Its first conditional learning control is now active, with no completed learning
result yet. A bounded actual-data GPU forward/backward passes;
a tiny FP32 normalization-order mismatch was diagnosed and corrected without
changing physical geometry or relaxing the parity gate. Existing forward and
the completed rehearsal driver are unchanged. No coherent-subject learning is
claimed yet.

The [12-subject coherent cohort](COHERENT_SUBJECT_COHORT_002_PROTOCOL_20260929.md)
finished from committed source `673b8fc`, session `9774` EXIT0, with every
predeclared plan accepted and its71-file completion inventory authenticated.
Eight training and four development subject maps retain their preassigned IDs.
Its separate section-generation stage68817 exited0; the section cohort and
target-capacity audit are frozen and readable. Every censored draw is retained.
The native training run consuming these data remains protected while active.

The [fresh image-key retrieval experiment](IMAGEKEY_RETRIEVAL_PROTOCOL_20260929.md)
completed4,000 updates, terminal8803 EXIT0, from `edc99b8`. Its independent
audit45886 EXIT0 passes integrity and every fixed candidate-stage gate.
Eligible group-macro normal error47.50->40.36deg and top32 physical-plane
capture10.59%->68.46% versus original003 at the matched4k budget. However,
full-frame corner capture within1mm is only4.28% at top32; this is not usable
alignment accuracy. [Full result, pins and limitations](IMAGEKEY_RETRIEVAL_001_RESULT_20260929.md).
Its whole fresh4k checkpoint is now used by the active conditional native ribbon
control, without encoder merging or imported old weights. The separate raw Allen
diagnostic has already failed adequate transfer; the active real+synthetic control
instead continues whole A6k after the negative dropout comparison. Neither path
substitutes for honest global capture or real final-test qualification.

Still required: convincing arbitrary-plane global and native joint accuracy,
usable optional constraints, genuine animal-disjoint validation, calibrated
joint uncertainty/site-region probabilities, frozen fair external comparison,
and practical integration/delivery in the existing desktop Anatomy Tracker.
