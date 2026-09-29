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
