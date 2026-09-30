# Current joint-model development — 2026-09-30

The native goal is active. No heartbeat automation is used. This is an unfinished
model-development goal, not a shipped or benchmark-qualified system. All work,
data, temporary files and runs belong on `I:`.

## Latest handoff

**2026-09-30 priority update:** the user's required model is direct probabilistic
coordinate prediction trained jointly with anatomical fitting, with fitting
performance improving future coordinate predictions. The old catalogue-centred
path is no longer the default next experiment. Both final v6 TRAIN diagnostics
are completed and compactly closed out. See
[direct joint-model design, results and staged training](DIRECT_JOINT_MODEL_20260930.md)
for the current work; the list below is historical evidence, not a fresh queue.

The whole-model run `I:/AnatomyTracker/runs/joint_v7_direct_joint_001` completed
6,000 updates and terminal session 17600 exited successfully. Fitting-only gradient
reached the direct pose head (norm 6.22668), but accuracy is not satisfactory:
eligible-subject macro TRAIN MAP normal error 14.87 degrees versus DEV 56.12;
center error 733 versus 2,900 um. Only 311 distinct eligible TRAIN planes were
repeated. This indicates substantial memorization, not a successful global model.
The focused known-pose fitter diagnostic worsened spatial error 55.67 -> 96.84 um,
despite preferring the correct over a perturbed plane in 31/32 cases. Preserve the
checkpoint and continue this same architecture with more distinct planes, real
images, direct physical pose supervision and anchored deformation training.

Full native synthetic/weak-real evaluation completed448 observations: synthetic
56.12deg/4,601um visible-grid error; real57.38deg/7,474um weak full-canvas affine
error. Output `runs/joint_v7_direct_joint_001_native_development_002`; not a
public benchmark. New GUI-consistent real192px preparation is complete (same
58TRAIN/6DEV donors), and the actual offscreen Qt inference/save/reload smoke
passed after fixing its stale overlay canvas size. No accuracy qualification.

TRAIN-only4,096-plane192px preparation finished in3748s. All eight subject
shards,4096 section-file hashes and inverse-map hashes passed the frozen audit;
the maximum sampled inverse-map interpolation error per subject was below.5um.
This is a reusable source map set, not a sufficiently large final training bank.

The fixed4,096-plane trainer queue **95614 was cancelled before launch (exit1)**.
Do not restart `training/train_joint_v7_expanded.py`. Its finite training bank was
too small as the intended main corpus. The remaining generator is useful because
it freezes reusable inverse anatomical maps, not because4,096 is the final scale.

`training/train_joint_v7_streaming.py` continued the same whole6,000-step model
and AdamW state with fresh physical sections every synthetic update. Its first
launch `_001` stopped before any update on a sampler unpacking typo; the
repaired `_002` exited cleanly after24,000 updates and144,000 eligible new synthetic
planes, with8113 real TRAIN images/58donors and the unchanged64 DEV/6donors.
This is an intermediate stage; larger-data collection proceeds alongside training.
DeepSlice generated~920k synthetic images (58k low-tissue exclusions) and used
131,240 slide-mounted plus442,680 S2P real images in its two-phase programme
([primary paper](https://www.nature.com/articles/s41467-023-41645-4)). Million-scale
synthetic exposure and hundreds of thousands of trustworthy real sections are
planning targets, not proof that size alone will meet the accuracy goal.

The [frozen streaming-stage result](STREAMING_JOINT_V7_STAGE_RESULT_20260930.md)
shows final direct MAP errors of54.73deg/12.06mm on legacy96px held-out
synthetic DEV versus3.61deg/657um against weak real DEV affines. This does not
qualify arbitrary-plane performance. A fixed192px rerender of the same held-out
synthetic subjects is [complete and audited](JOINT_V7_MATCHED_192_DEV_RESULT_20260930.md):
direct MAP error remains8.2–8.7mm/38–40deg by background mode. The fitter
changed no eligible branch and does not update full pose at inference. The next
architecture experiment [now has a frozen negative result](JOINT_V8_FEEDBACK_PILOT_RESULT_20260930.md):
the corrected loop improves the prior branch by143um on average but its
candidate ranking loses287um and selected error remains about9mm. The next
priority is independent anatomy diversity and richer candidate supervision,
not blind scaling of this deficient checkpoint; no public benchmark.

The streaming sampler has512 fixed global-affine variants of8 base synthetic
subjects from one atlas; these are not512 biological animals or independently
varying local morphologies. Each generated plane has one background mode; retries,
used planes and complete replay parameters are preserved separately. Map targets
are explicitly approximate. Capacity statistics record director targets outside
the current fitter's component bound without clipping the truth. The first64
fixed diagnostic planes took.65s; training step1 peaked at5038MiB allocated
GPU memory and fitting sent a nonzero gradient to the direct pose head. At1000
updates, weak real DEV normal error was4.57deg but arbitrary-plane synthetic
DEV remained59.68deg and13.13mm five-point error, worse than its step0
coordinate error. This is not usable arbitrary-plane accuracy.

The8113 real expansion exited and its frozen `completed.json` reports6833 new
downloads with zero failures, plus1280 exact reused TRAIN images and64 exact
reused DEV images. Independent rehashing matched every declared output hash;
the8177 records have unique section IDs,58 TRAIN and6 DEV donors with zero
overlap, finite geometry and the declared192px float16 array shape. These are
weak Allen affine labels, not expert-verified alignments; the extra images add
no new donors.
Large-cohort TRAIN-only metadata acquisition is frozen at
`data/joint_v7_reserved_train_metadata_001`:1885 reserved donors and263754
candidate sections. Independent audit matched all raw and summary hashes, found
zero reserved-split leakage or duplicate section IDs, and found finite nonzero
2D/3D transform determinants. Donor-sharded TRAIN-only image acquisition has
started separately; it reuses8113 exact matching TRAIN source JPEGs and never
uses the64 DEV images. Its output is protected while running.
Calibration/final donor reservations exist before new image access; historical
exposure caveats remain explicit. These upstream affines remain weak labels.

Readouts are saved at step0,1000 and every4000: likelihood loss, normal/centre
errors and reflection-aware observed-coordinate errors. Real DEV192 is matched
in resolution; legacy synthetic DEV96 is not, so its train/dev gap is not pure
generalization evidence. Real and synthetic canonical charts also differ near
dominant-normal seams; observed pixel geometry remains explicit and unchanged.
All constraints remain absent in this stage. The optional-constraint packer
encodes elevation bounds and depth in context slots7–11, while the model's
context docstring describes a unit probe direction/cone in those slots; this
semantic mismatch must be resolved before constraint training or GUI wiring.
The native goal remains active. The user permits selective useful agents but
wants to avoid excessive delegation, checks and status updates.

Next-stage probe-observation preparation is implemented separately in
`training/arbitrary_plane_probe_observations_v7.py`; it is **not yet trained or
wired into the GUI**. One probe is sampled independently of section geometry in
subject space, then intersected with each finite slab. Clicks retain their
through-plane offsets; a straight subject probe is not forced straight in CCF.
Entry-disk, elevation-interval and maximum-depth observations include missing
and contradictory variants, with no invented azimuth or AP-range anchor.
The corrected CPU check `data/joint_v7_probe_observation_cpu_check_002` passed
on one fixed TRAIN probe/64 sections (3 intersections, 6 retained clicks across
presentations; maximum frozen-raster interpolation discrepancy 0.508um).
Check001 is superseded: its voxel-centre-origin convention was corrected to
the project's voxel-boundary origin before any training use. The forthcoming
streaming trainer remains constraint-free.

`training.evaluate_joint_v7_expanded` is a prepared but unrun evaluator for the
cancelled fixed-bank trainer. Adapt its input/output paths to the streamed run
after training; do not launch it unchanged. It scores all16 fitted branches on
the same384 synthetic DEV96 observations plus64 real DEV192 images. It preserves
raw predictions and separates selected, prior-selected, unwarped and diagnostic
oracle errors. Real192 vs the original96px experiment is not a matched comparison;
the trainer's own step0/final192 readouts provide the matched direct-pose check.

GUI integration now includes explicit animal/specimen/experiment/section IDs,
rigid native-pose adjustment and exact restore of the model prediction. Native
probe trajectory pooling/mapping/export requires a single explicit animal across
all contributing sections and rejects conflicting supplied specimen IDs; it
never silently drops another animal. Legacy-only workspaces remain unchanged.
Manual edits preserve original model arrays and invalidate their uncertainty
interpretation. The actual offscreen Qt workflow/ID-conflict checks passed in
`runs/joint_v7_gui_identity_pose_smoke_004`; this verifies interaction and
coordinate persistence, not model accuracy or uncertainty calibration.

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
- Full-coverage runner96942 from `2935584` and independent CPUaudit86936
  exited0. Whole C10k→A22576 completed12,576 updates and passed all12 fixed
  gates; conditional B was not run, so no mining-effect contrast exists.
  Real-development plane capture32 rises90.61%→96.82%, normal12.43°→9.29°,
  but the TRAIN-only image-free normal prior scores4.17° and selected finite-frame
  error worsens3.353→3.460mm. Synthetic eligible normal remains37.46° with
  capture32 77.88%; this is a candidate-stage pass, not usable joint alignment.
  New generated attempts50,304/eligible50,035 complete98,304 attempted lineage
  cells, not98,304 successfully supervised positives. Output is frozen/readable.
  Audit SHA256 `c75c4ff817e7603eca31c8086080aafdb7f0953b3fec58c3f3f2f7842cf2afb3`.
  [Audited result](COARSE_FULL_COVERAGE_RANKING_001_RESULT_20260929.md).
  TRAIN-only saved-candidate analysis86237 from `a35038e` also exited0.
  95.15% synthetic/100% real eligible top1 starts exceed the old local teacher
  generation box. Even oracle best-frame/R starts exceed it83.63%/100%; real
  median best-frame tangent displacement1.436mm and log-V-span correction.16164
  show a finite-frame training gap. Coarse predicted R differs from best-frame R
  on46.72% of eligible real top1 rows; keep both reflected hypotheses.
  [TRAIN capture evidence](TRAIN_CAPTURE_RESIDUALS_001_RESULT_20260929.md).
- Actual coherent TRAIN candidate cache97190 from `6a172de` and independent
  CPU reconstruction75081 exited0. All1,920 observations/640 sections/eight
  synthetic subjects are retained (1,149 eligible,771 censored), without DEV
  section reads. Eligible top128 plane capture is only39.65–51.81%, top1
  normal error48.40–53.27 degrees, and predicted-R frame RMS11.14–13.25mm.
  Even oracle best beam/frame RMS is3.54–4.85mm. Coarse retained mass8.43–10.88%
  is uncalibrated, not an exhausted posterior. Previous fixed12mm synthetic
  capture cannot be presumed to transfer to coherent acquisition canvases;
  broaden actual coarse training as well as native capture before qualification.
  A saved-frame necessary-reach check finds78.78–79.33% of eligible base
  coherent sections exceed the three-step log-span reach from12mm. None of
  the acquisition frames exceeds that span bound; it is not a full diagnosis
  of retrieval failure. Change future capture support, not current live settings.
  [Audited TRAIN-only result](COHERENT_TRAIN_CANDIDATES_001_RESULT_20260929.md).
- The independent native mark adapter is implemented as `raw_marks_to_ribbon_ccf`
  in `training/arbitrary_plane_ribbon_marks.py`, committed `f5cdf77`. It maps
  recorded raw-to-model coordinates onto each retained observed curved surface,
  preserves cell/reflection alternatives and invalidates out-of-FOV marks.
  One completed TRAIN prediction passed direct CPU interpolation checks, but
  there is no GUI wiring, electrode accuracy or probability-calibration claim.
  [Remaining integration gaps](JOINT_CONSTRAINT_INTEGRATION_GAPS_20260929.md).
- Earlier bridge, native-no-probe, signed-pose and coarse-rehearsal outputs/audits
  are frozen and readable. TRAIN-only pose-cost-direction diagnostic30641 also
  exited0. Full-coverage and normalized signed-pose outputs are frozen/readable;
  the latter's independent audit also completed and confirms a failed normal gate.
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
One subsequent scheduled TRAIN B4 forward (92025 EXIT0) found signed
projection/base-evidence RMS1.532–1.794%, with raw per-axis RMS
(.08283151464815028,.0678978954706466,.07608602924318171). No optimizer or
model change was used for that measurement.
[Activation result](SIGNED_POSE_TRAIN_ACTIVATION_001_RESULT_20260929.md).
The next fixed control divides the existing signed maps by those TRAIN-only
axis RMS values; no new architecture/features/loss or scale sweep. The optional
three-value constructor argument is implemented with no parameter/RNG changes
and an unchanged `None` path. Compatibility preflight17128 from `e6c97e6`
passed exact zero-head outputs/common parameters/RNG and finite backward,
with no optimizer update; receipt SHA256
`1d87df5a33bd2da6df570a84108b9e9a5b0fec888ef10f4c52578b8f7f4d68c4`.
Matched training90795 from `2a86507` exited0 after2,000 updates. Its completed
independent audit passes integrity and fails the unchanged oracle-normal20% gate.
Correct-reflection centre improves709.931→644.067um versus the unscaled control,
but selected centre worsens842.274→960.625um and normal improves only
6.10189→5.87902degrees. No promotion, extension or scalar sweep. All artifacts
are now frozen/readable. [Completed result and exact bindings](NORMALIZED_SIGNED_POSE_EVIDENCE_001_RESULT_20260930.md).
Coarse session96942 has exited and its whole-A endpoint was audited;
do not splice its encoder with this separately trained native control.
[Prospective protocol](NORMALIZED_SIGNED_POSE_EVIDENCE_PROTOCOL_20260929.md).
The failed, independently audited normal-learning gate now triggers the
enabled TRAIN-only feature-Jacobian/readout comparison
in `training/diagnose_joint_v6_train_pose_jacobian_readout.py` (`634b27c`). It
uses one completed whole signed checkpoint, exact24 prior TRAIN observations,
fixed oracle fields/reflection/PSF and three paired learned-versus-damped-solve
updates; no optimizer or hyperparameter sweep. Root and independent source
review found no correctness blocker. It is off-policy, not qualification, and
compares against the unscaled head, not the latest normalized readout.
[Fixed protocol and interpretation limits](TRAIN_POSE_JACOBIAN_READOUT_DISCRIMINATOR_20260929.md).
Do not infer multiaxis or learned-deformation success from the oracle diagnostic,
or infer that amplitude imbalance alone proves a scaling fix.
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
