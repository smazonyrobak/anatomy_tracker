# 121 joint pose–mapping training protocol (predeclared)

## Question and lineage

Does updating the **same** arbitrary-plane image encoder, probabilistic pose heads,
atlas-aware full-frame correction, and atlas-conditioned local map let anatomical
fitting improve blind coordinate prediction? The parent is this project's randomly
initialized 111 step-1959 checkpoint. The 120 step-256 matcher is **not** loaded:
its atlas-aware and support-only arms were essentially identical on synthetic DEV.
The 120 matcher and 083 correspondence matcher are freshly initialized here; no
external weights, pseudolabels, final-test animals, or public benchmark data enter
training. This is a development experiment, not a claim of deployment readiness.

## Data and schedule

- 8,000 optimizer updates, each accumulating gradients from two **independently
  drawn eligible** v3 TRAIN synthetic sections (16,000 presentations). The sampler
  independently randomizes each plane, framing, deformation, damage, appearance,
  and background. It does not create deliberately paired backgrounds or reject a
  chance repeat. Virtual deformation bases are not independent biological animals.
- Each sample retains its generator physical-section identity, draw seed, source
  provenance, exact warped observed-pixel→CCF field, valid-tissue mask, finite-slab
  PSF, and observed-image **affine gauge** state. The affine gauge is fitted to the
  warped field; it is **not** the pristine pre-warp cutting plane. Direct pose is
  supervised against this gauge, while local mapping is supervised against the
  warped pixelwise CCF field. This prevents the pose head from being asked to
  explain non-affine tissue deformation.
- The previously frozen DEV panel retains an older, inaccurate prose value for
  `pose_gauge`; its pixel arrays and target hashes are unchanged. Evaluation
  interprets `target_state` according to the actual affine-fit generator code.
- At every eighth update after 2,000, one reserved TRAIN coronal and one reserved
  TRAIN sagittal Allen image add a weak-affine **pose-only** retention term. Their
  inherited Allen alignment is not expert ground truth and never supervises the
  local map, atlas fitting, calibration, or the synthetic DEV gate. Donor/specimen/
  experiment/section IDs and source hashes are retained.
- Checkpoints: 0, 2,000, 5,000, 8,000 updates. A separate frozen, donor-disjoint
  synthetic/real development evaluator chooses a checkpoint; the 8,000-step
  checkpoint is not automatically promoted. No full public DeepSlice benchmarking.

## Architecture and blind actions

The 111 shared encoder/lateral, direct pose/anchor logits and states, and
atlas-conditioned `warp_shared`, `warp_condition`, `atlas_encoder`, `pair`, and
`warp` train together with a fresh 120 global matcher. Fresh 083 image–atlas
correspondence embeddings supply a **training-only auxiliary CE loss** on shared
image features; 083 is not part of the inference mapping or scoring path. The
atlas-conditioned mapper already computes local image–atlas correlations. The
earlier 089 two-pass 083 integration failed to improve blind selected error, so
this run does not reintroduce that extra inference loop without evidence. Legacy
rank/uncertainty heads remain frozen; uncertainty
outputs are **uncalibrated**. The original no-shift candidate remains available.

Every synthetic image starts with the frozen rule for 14 blind candidates (prior
top eight base and top six anchors) plus two diverse unused anchors. Neither the
truth state nor the warped CCF field enters this blind proposal or inference score.
The score compares original and corrected actions over the same 16 candidates.
For **training only**, the observed-affine-gauge exact and locally perturbed states
are appended to learn near-pose mapping and the no-shift gate. The physically best
blind candidate and a physically wrong blind candidate are selected using synthetic
truth solely to form losses; they are not injected into the inference beam.

Fit feedback is applied only when the corrected best blind candidate is within
**1.5 mm** of the observed-affine-gauge state; the bounded local map must not be
asked to rescue gross pose errors. The wrong candidate contributes a fit contrast
only if it is more than 1 mm worse in rigid-gauge error yet has atlas coverage
within 0.10 of the best candidate after mapping. Otherwise the contrast is
skipped and counted. No correct-map
target is applied to the wrong candidate; a wrong plane must not be repaired by
twisting the local tissue map. The learned local displacement is affine-free and
bounded; global placement must reside in the pose.

## Loss and feedback path

With distances in mm, the synthetic loss per image is:

`L = Ldirect + 0.5 Laction-KL + 0.25 E[action rigid cost]
   + Lmap + 0.5 Lselected-rigid + 0.05 (CEfine + CEcoarse)
   + 0.03 Lwarp + λfit (Lfit + 0.5 Lcoverage + Lcontrast)
   + 0.1 Lexact-no-shift + 0.1 Lnear-shift`.

`Ldirect` is the 111-style full-160 mode/reflection pose-mixture objective but
measures the affine gauge at original valid pixels and fixed frame points, **not**
the warped CCF field. The 16 blind original/corrected action target is their
rigid-gauge physical cost; it is detached only for the score distribution, not for
the supervised geometry. `Lmap` is sampled original-pixel CCF error for exact,
near, and a rigid-error-gated best blind candidate. `Lselected-rigid` penalizes
the corresponding corrected full-frame errors. Fine/coarse 083 CE uses only valid
known-CCF synthetic correspondences from exact/near poses. Warp regularization
uses displacement magnitude and spatial roughness; it is not a substitute for
anatomical agreement. `Lfit` is normalized, polarity-invariant local image–atlas
correlation on tissue with finite-through-plane PSF rendering. Separate coverage
prevents a zero-overlap plane from obtaining a spuriously perfect fit.
`Lcontrast = relu(0.10 + fit_best - stopgrad(fit_wrong))` on support-matched pairs;
the wrong branch is not deliberately degraded. The exact gate favours no correction
and the near gate favours a correction. `λfit=0` for updates 1–2,000 and ramps to
at most 0.05 over updates 2,001–3,000. A single measured gradient-ratio cap may
reduce it so fitting does not overwhelm the synthetic geometric targets.

The critical path is **fit loss → mapped coordinates → differentiable finite-slab
atlas render and local map → corrected state → direct pose state / shared encoder**.
At the first eligible fit-enabled TRAIN sample, log a focused fit-only autograd
audit on corrected state, direct pose state, one pose head, shared encoder, and
matcher. Log the fit-only/direct state-gradient ratio and its capped weighted
value: a merely nonzero but negligible gradient is not evidence of useful
feedback. A nonzero matcher gradient alone is insufficient. If fit-only gradient
cannot reach direct pose and encoder on a sample with common support, the joint
learning claim fails and the run must be fixed before continuation.

Real-image retention contributes `0.10 ×` its own weak-affine full-bank pose loss
on the scheduled updates only. It does not have synthetic map/fit losses.

## Evaluation and decisions

The frozen evaluator compares checkpoint 0/2,000/5,000/8,000 and parent 111 on
the same DEV sections using the identical blind 14+2 **selection algorithm**;
each checkpoint generates its own candidate states and hence its own beam contents.
Report selected and
best-available rigid-gauge error, warped-site mapping error, original-vs-corrected
selection, no-shift action frequency, fit/coverage/warp by correct and wrong
candidate, exact/near oracle-given mapping, angle/background/artifact strata, and
the six-donor/64-image coronal and separate eight-donor/158-image sagittal
real weak-affine guardrails, contingent on donor-disjointness checks. Show distributions and
failures, not only means. A lower TRAIN fit loss by itself is not success.

An advancement candidate must improve blind selected rigid error by at least
0.30 mm against parent 111 **and** 121 step 0, improve blind mapped-site error
by at least 0.20 mm, retain exact/near mapping, and not worsen either real plane
family's weak-affine selected mean by more than 0.20 mm. These are development
engineering gates, not statistical proof or final shipping criteria. Report even
if they fail; do not relabel a weak improvement as deployment readiness.

Unseen-animal expert validation, probability calibration, electrode uncertainty,
existing-GUI integration, and a fair locked DeepSlice comparison remain required
after strong internal evidence. The synthetic sections and weak Allen references
cannot establish those claims.
