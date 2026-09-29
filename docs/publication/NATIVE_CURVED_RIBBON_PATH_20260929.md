# Opt-in native curved-ribbon refinement

`ArbitraryPlaneJointModelV6(..., ribbon_deformation=True).refine_ribbon(...)`
adds conditional physical-surface refinement to the same joint model. It uses
the existing histology/atlas encoders, pair evidence, shared GRU, bounded 9D pose
update and final score head. One new zero-initialized hidden-to-six-channel
convolution predicts the ribbon fields. Its CPU RNG initialization is isolated;
the default constructor has no new parameter keys and default `forward` is
unchanged. The old 2D decoder and uncertainty heads remain unused in this method.

Inputs are image/outline/availability, an atlas volume, initial states B,K,12,
initial component log masses B,K,R, horizontal-reflection flags R, output raster
shape, physical atlas origin/spacing, fixed support origin, known PSF offsets
and weights S or B,S, and a step count. Initial states are copied to R separate
hypotheses. Neither hidden states, poses nor deformation fields are averaged
across reflection. No global retrieval or fabricated tail mass is supplied.
Each sample must supply at least one finite component log mass; an all-excluded
set is not a valid conditional distribution. Atlas tensors retain the existing
FP32/FP64 rendering contract.

At iterations 0 through T, the previous hidden map predicts an **absolute** field
(zero during the fixed pose-only prefix). After upsampling, horizontal branches
spatially flip the six-channel map into the canonical raster. Channels 0:3 are
local displacement in µm, bounded before projection by tanh×200; channels 3:6
are dimensionless local director changes bounded by tanh×0.2. The ribbon
primitive removes the surface's full-canvas affine gauge and jointly rescales
surface/director fields to its 0.35 physical derivative bound. Geometry runs
outside autocast at least FP32; no coordinate gradients are detached.
These amplitude limits are design choices, not established coverage bounds for
the coherent-subject targets.

The resulting physical PSF grid is spatially flipped back into each observed
raster. CCF components and director signs are never negated by this operation.
The atlas is sampled directly along this curved finite-thickness grid, not by
warping a previously sampled 2D atlas image. The shared GRU consumes the new
render at every iteration. Optional seven-channel coordinate evidence uses the
**actual observed centre surface** at pixels 0,4,..., plus observed x/y and
reflection signs, rather than a planar proxy. Each of the first T GRU outputs
updates that hypothesis's own pose; the last iteration re-renders its final
pose and decoded field before scoring.

State and constrained-field sequences have B,K,R,T+1 leading axes; observed
centre sequences are supplied for physical-coordinate supervision. Final
observed slab/centre coordinates, renders, updates and derivative-limit
diagnostics remain differentiable. No old total-2D-SVF target or 35D covariance
interpretation applies. First-stage section processing is identity and PSF is
known; noisy acquired constraints and GUI integration are not wired here.

Kept-component weights are softmax(initial log mass + **final score only**).
Intermediate image evidence is not repeatedly multiplied into probabilities.
These are explicitly uncalibrated conditional weights over supplied components,
not complete-gallery probabilities, retained global mass or electrode-region
confidence. The field head is untrained. This commit has AST/source inspection
only: no model execution, GPU run, gradient check or accuracy claim. Independent
source review found no blocker; root review precedes any experiment. NumPy target and active training
drivers are unchanged.

## Actual-data GPU execution, after rehearsal exit

One fresh whole random model was exercised on the frozen eligible reflected
section2, imperfect-brush observation, with two independent reflection branches
and two recurrent updates. It used the fitted canonical frame as an oracle
start, known PSF, and zero ribbon-head initialization. This is one graph check,
not learning, global capture or point-estimate accuracy.

The first check failed a strict1e-6 zero-deformation render-parity threshold.
The diagnostic repeat preserved that failure: maximum intensity difference
1.58548e-5, RMS3.14479e-7. On identical physical queries, tensor-vector division
versus the older renderer's per-axis scalar division changed normalized FP32
grid coordinates by2.38419e-7. Re-rendering with the scalar arithmetic matched
the planar renderer exactly; no axis, reflection, PSF or physical-coordinate
discrepancy was found. The second run still exited1 after recording its failed
parity gate, although its backward checks passed.

The coordinate-grid renderer now uses the same per-axis scalar normalization
as the existing planar renderer. The third run exited0 with the original
threshold unchanged: direct-grid rerender error0, zero-field planar-render
error0, initial reflection-permutation error0. Physical coordinates and PSF
weights are unchanged. Existing frozen data and failed diagnostics are retained;
future rendering carries the revised source hash.

Finite, nonzero gradients reach the ribbon and pose heads, slice/atlas/shared
encoders, score head and coordinate input. Forward0.737s, backward0.122s and
peak allocated1.887GB were observed on RTX2080Ti, excluding atlas loading.
No optimizer step was applied. The two final branch states remain distinct.
Nonzero trained fields, active bound clipping, learned accuracy and calibrated
probabilities are not established by this zero-head execution.

Outputs under `I:/AnatomyTracker/runs/`:

- `joint_v6_native_ribbon_gpu_check_001`: original failed check, no completion.
- `joint_v6_native_ribbon_gpu_check_002`: diagnosed parity failure and completed
  backward; completion SHA256
  `f84ee71bd7ac5d340e540440e4f10ee5d11b7a8a7935129630992aff8ba0aa47`.
- `joint_v6_native_ribbon_gpu_check_003`: corrected arithmetic, full check pass;
  completion SHA256
  `04489d73591441e5a6634945e5ee79714885b45629c1347bcd8e7b6ad37f6a15`.

Each retained diagnostic records its actual script/source, input identities and
raw outputs. Run002's completion filename does not supersede its failed gate.
