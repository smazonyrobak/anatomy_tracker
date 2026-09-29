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
- Real+synthetic001 and independent audit11096 both exited0. Whole A6k→8k
  continuation fails both real improvement gates and imperfect-brush synthetic
  angle retention. Real normal38.59→42.99deg, capture32 .13939→.18333;
  low sampled NCE does not establish global fitting. Do not promote or extend.
  [Result and exact bindings](IMAGEKEY_REAL_SYNTHETIC_001_RESULT_20260929.md).
- Training-only exact-anchor/gallery diagnosis84429 from `f57e399` exited0.
  On all256 training images/58donors, failed8k achieves96.03% near-anchor hit1
  but22.59% full-gallery plane capture32. No acquisition frame has a catalogue
  candidate within1mm finite-frame/10deg tolerance. Tangent centre and span
  contribute about80% of nearest-frame squared error. Empty references alone
  do not explain the gap. This supports a chart-transfer problem, not a claim
  that geometry conventions are broken or that unseen-donor overfit explains
  everything. [Reconstructed result](REAL_TRAINING_RETRIEVAL_001_RESULT_20260929.md).
- Next is a matched-budget continuous-plane chart bridge: matched affine
  positives versus a50/50 mixture of matched and catalogue-centred12mm
  positives, preserving source roll. Both arms share explicit weak-reference
  eligibility and physical-plane negative exclusion. Same wholeA6k parent,
  2,000 updates each, fixed endpoints; no public benchmark or encoder merging.
  [Protocol](CANONICAL_CHART_BRIDGE_001_PROTOCOL_20260929.md). Runner96952 from
  `ca1cb38` and independent audit41128 both exited0. Real development normal
  error A38.42→B9.54deg and capture32 A19.85%→B92.12% show a large paired gain.
  B nevertheless fails the whole gate: no-brush synthetic capture decreases
  2.25points versus the parent, beyond the fixed2point allowance. No waiver.
  [Audited result](CANONICAL_CHART_BRIDGE_001_RESULT_20260929.md).
- Next coarse step is a separately declared whole-B experimental continuation
  with lower learning rate and stronger broad synthetic rehearsal, preserving
  the canonical real positives. B is an experimental warm start, not a promoted
  model; the six development donors are not untouched final validation.
  The reviewed [rehearsal protocol](CANONICAL_REHEARSAL_001_PROTOCOL_20260929.md)
  fixes2,000 updates,LR.00025 and2/3 synthetic loss, with original retention gates.
  Rehearsal91005 and CPUaudit43867 exited0. Integrity passes, whole gate fails:
  eligible synthetic capture32 improves72.52%→78.14%, but real normal error
  worsens9.54°→12.43°, exceeding the fixed+2° retention limit. Do not promote or
  start a scalar LR/replay sweep. [Audited result](CANONICAL_REHEARSAL_001_RESULT_20260929.md).
- Native conditional pose+ribbon001 (89162, `dc32605`) and independent audit84435
  both exited0. Oracle centre error814.24→716.56um improves only about12%, below
  the fixed20% requirement; plane-normal error6.138→6.103deg barely changes.
  Selected error2685.48→838.82um is helped strongly by reflection resolution.
  Do not promote this truth-near, known-PSF control to honest global inference.
  Revise local plane learning before a retrieved-beam qualification experiment.
  [Full result and independent coordinate attribution](NATIVE_RIBBON_LOCAL_001_RESULT_20260929.md).
- The next matched native control adds six signed out-of-plane render-cost maps
  to the existing shared updater. Original whole4k parent, losses, sampling and
  training budget stay fixed; an explicit20% oracle-normal reduction is required
  alongside the original dense-coordinate gate. Implementation is committed as
  `0e93978`; one fixed TRAIN-batch preflight27377 exited0. Zero-head outputs and
  all original parameters/RNG match exactly, gradients pass, no optimizer applied.
  Receipt SHA-256 `4e5a2fd3a9697da6a87c435abf35f337bb383e596c46abaa2cc2560e0a39a76e`.
  Native signed-pose session70497 from `0e6dc0d` and independent CPUaudit4417
  both exited0. Integrity passes, scientific gate fails: correct-reflection
  centre814.236→709.931um improves12.81%, while normal6.13834→6.10189deg
  improves only.594%. Relative to no probes, normal improves just.000935deg;
  the extra signed-cost channels have not solved plane learning. Do not promote
  or splice this separate controlled refiner into coarse C.
  [Audited result](NATIVE_SIGNED_POSE_EVIDENCE_001_RESULT_20260929.md).
  [Predeclared protocol](NATIVE_SIGNED_POSE_EVIDENCE_PROTOCOL_20260929.md).
- Same-model full-gallery image-key to native-ribbon connectivity preflight62858
  exited0 from `d0eef25`. One original TRAIN observation592 passed K4/R2/T3
  chunk1-versus4 agreement; largest final coordinate difference.00293um.
  Coarse retained mass.0068891 and omitted mass.9931111 are reported explicitly;
  the refined tail remains unknown and probabilities uncalibrated. This uses
  the whole failed native control, not a qualified global model or merged
  refiner. No accuracy evaluation was performed.
  [Result and frozen bindings](IMAGEKEY_RIBBON_INFERENCE_PREFLIGHT_001_RESULT_20260929.md).
- Acquisition-view generator13699 from `4e89e3c` exited0; the independent CPU
  audit passed all128 TRAIN physical views/384 paired observations. Eligible
  raw/black/brush counts are76/76/75, totaling227/384; all censored rows remain.
  These reuse eight existing synthetic subjects, not new biological animals;
  target-capacity checks establish neither model performance nor generalization.
  [Result and exact audit bindings](COHERENT_ACQUISITION_VIEWS_001_RESULT_20260929.md).
  Fixed1,024 additional TRAIN real-image acquisition91765 from `d807454` exited0.
  Its independent audit passed the exact1,280-image union from58 TRAIN donors,
  untouched original256 links, byte hashes and coordinate transforms, excluding
  all six development donors. This expands sections, not biological animals.
  [Audited real expansion](ALLEN_REAL_TRAINING_EXPANSION_20260929_RESULT.md).
- The fixed full-coverage uniform-first experiment is now running as session96942/
  PID22144 from `2935584` in the separate, existing
  `I:/AnatomyTracker/agent_worktrees/coarse_retrieval_scaling` checkout.
  Arm A has passed1,900 of12,576 additional updates from the whole C10k parent.
  Fresh generated cells cover the remaining50,304 previously unattempted cells;
  real training uses the audited1,280-image union. Conditional arm B uses the
  same whole parent/budget/schedules with snapshot-hard negatives only if A
  fails the predeclared endpoint gate. No interim development evaluation.
  That checkout and the entire output tree remain frozen/protected until the
  whole runner exits. A guarded independent auditor is prepared, not executed.
  [Frozen protocol](COARSE_FULL_COVERAGE_RANKING_PROTOCOL_20260929.md).
- Earlier bridge, native-no-probe, signed-pose and coarse-rehearsal outputs/audits
  are frozen and readable. TRAIN-only pose-cost-direction diagnostic30641 also
  exited0; only the full-coverage run's output remains protected until its exit.
  Concurrent run timings are not
  hardware benchmarks. No independently trained encoders are
  merged; a future successful real warm start needs sequential native training
  of that **whole model**, not splicing in a separately trained native control.
- No heartbeat or additional user prompt is needed for these native-goal steps.

## Next whole-model integration decision

The existing public `forward()` still uses the older parametric proposal path.
The new functional `imagekey_ribbon_inference_v6()` now connects complete
image-key scoring, stable retained cells with separate reflection masses,
chunked same-model ribbon refinement and final image-based selection; one
actual TRAIN-row numerical preflight has passed. It is not yet wired into the
public forward or GUI, and supplies no new accuracy evidence. Original full
component mass plus each final refinement score is normalized once across all
retained cells/representations. Keep omitted coarse mass separate: normalized
retained scores are not calibrated global probabilities, and the refined tail
remains unresolved.

Continue one whole coarse checkpoint through native training; do not import
the signed-control refiner trained against its different4k encoder. If the
signed mechanism succeeds, first bootstrap that whole model's native components
with retrieval frozen, then introduce actual predicted TRAIN catalogue starts
and competing-cell selection. Truth-near warm starts or teacher-injected training
states must never leak into the reported inference evaluation. Derive the
capture curriculum from TRAIN residuals, not development-case selection.

The real bridge's roughly3.09mm MAP finite-frame error remains much larger than
the roughly.81mm native initializer error. Existing finite rasters can only
support interpolation-consistent acquisition crops within their observed FOV;
padding is unknown, not measured black tissue exterior. For larger shifted FOVs,
re-render existing frozen TRAIN subject plans through the exact subject map and
PSF. This creates new acquisition views, not independent animals. Transform all
coordinate/visibility fields consistently, apply raster reflection once, and
refit the curved surface's canonical pose gauge after changing its canvas.

Signed evidence failed. Use a bounded TRAIN-only probe-direction diagnostic
to decide between adapting the existing shared features with retrieval rehearsal
and changing the update readout. Do not infer that choice from unfinished runs
or start full-gallery native training blindly. The coarse-rehearsal auditor has
completed, as has the signed-pose auditor. The prepared diagnostic is now enabled
against failed audit SHA256
`7acdc3ef87f85691b296f67fc828428aa090b1afed88a5ce05853a6e81853ad5`
and completed as30641. It uses only eight TRAIN sections/all paired presentations,
exact oracle fields/reflection and the original whole4k features: no optimizer,
development data or learned-update qualification. Full-map cost directions chose
geometry-improving probes128/144 times (oracle support weighting137/144), so
these favorable oracle conditions do not support dismissing the frozen features
as directionless. Independent raw-map remeasurement confirms those counts and
finds no≤1e-6 cost ties; median absolute full-map plus/minus difference.02503.
[Result](TRAIN_POSE_COST_DIRECTION_001_RESULT_20260929.md).
Next inspect signed evidence scale on one actual scheduled
TRAIN batch before changing the update readout; do not infer multiaxis or
learned-deformation success from the oracle diagnostic.
The exact TRAIN-only acquisition-view extension in
[this protocol](COHERENT_ACQUISITION_VIEWS_001_PROTOCOL_20260929.md) has completed
all128 planned views and passed its frozen-corpus audit:227/384 paired
observations are support-eligible and all exact target representatives fit the
current geometric caps. These reused synthetic groups are not new animals.

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
Its first conditional learning control is completed and audited but fails the
oracle coordinate-improvement gate. A bounded actual-data GPU forward/backward passes;
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
The first native training run consuming these data has also exited and is frozen.

The [fresh image-key retrieval experiment](IMAGEKEY_RETRIEVAL_PROTOCOL_20260929.md)
completed4,000 updates, terminal8803 EXIT0, from `edc99b8`. Its independent
audit45886 EXIT0 passes integrity and every fixed candidate-stage gate.
Eligible group-macro normal error47.50->40.36deg and top32 physical-plane
capture10.59%->68.46% versus original003 at the matched4k budget. However,
full-frame corner capture within1mm is only4.28% at top32; this is not usable
alignment accuracy. [Full result, pins and limitations](IMAGEKEY_RETRIEVAL_001_RESULT_20260929.md).
Its whole fresh4k checkpoint was used by the completed conditional native ribbon
control, without encoder merging or imported old weights. The separate raw Allen
diagnostic has already failed adequate transfer; the completed real+synthetic
continuation from whole A6k also failed and is not promoted. Neither path
substitutes for honest global capture or real final-test qualification.

Still required: convincing arbitrary-plane global and native joint accuracy,
usable optional constraints, genuine animal-disjoint validation, calibrated
joint uncertainty/site-region probabilities, frozen fair external comparison,
and practical integration/delivery in the existing desktop Anatomy Tracker.
