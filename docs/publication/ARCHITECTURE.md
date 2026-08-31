# Proposed unified architecture

## Design decision

The release candidate is trained from random initialization as one model with
one optimizer and one checkpoint. It has no learned dependency on AtlasPose,
AtlasWarp, DeepSlice, ImageNet weights or any earlier project checkpoint.
Those systems remain frozen comparators only. The first warm-start joint
implementation is retained as a systems prototype and development baseline;
it is ineligible for final architecture selection.

The leading hypothesis is a compact recurrent correlation pyramid. A shallow
histology stem produces reusable multiscale structural features and a coarse
probabilistic AP/L--R/D--V pose. The current CCF plane is rendered, encoded in
the same learned structural space, and compared with the slice through local
multiscale cost volumes. A real shared-weight ConvGRU state then emits a pose
increment, in-plane similarity increment, compatibility energy, matchability
mask and residual stationary velocity field. Updating the pose causes a fresh
atlas render before the next iteration. Global plane pose and bounded local
deformation remain explicit even though they are optimized jointly.

Deployment uses three ONNX entry graphs exported from that same checkpoint: a
source initializer/encoder, a cached coarse candidate scorer, and a final dense
refiner. The first graph computes the source pyramid once; the scorer reuses
its coarse levels across pose candidates and recurrent updates without
constructing dense maps; the final graph decodes one pose-bound deformation.
A deterministic host-side CCF renderer runs between calls. These are entry
points into one learned model, not separately trained models. The split avoids
re-encoding the same section or integrating unused deformation fields for every
wrong-plane candidate.

The design is provisional until the matched cold-start architecture screen is
complete. This document defines the leading hypothesis, not a completed or
selected model.

## Inputs and outputs

Inputs:

- grayscale histology section in the orientation selected by the user;
- optional tissue outline, obtained automatically in automatic mode or supplied by the smart brush in assisted mode;
- an explicit outline-availability indicator, so absence of a mask is not confused with a section that fills the canvas;
- differentiably rendered Allen CCF template/annotation planes at the current pose;
- recurrent state from the preceding refinement step.

Outputs:

- posterior or scored candidate representation for AP, L--R tilt and D--V tilt;
- residual global pose update;
- residual 2-D similarity transform for nuisance rotation, scale and translation;
- stationary velocity field integrated into forward and inverse dense maps;
- visible/matchable-tissue estimate if justified by development evidence;
- slice--plane compatibility energy and its monotone risk score;
- intermediate outputs at every refinement iteration for deep supervision and audit.

## Recurrent render--compare--correct loop

1. Canonicalize the histology view and predict an initial atlas-plane distribution.
2. Render the CCF plane at the current pose.
3. Preserve full-plane metric geometry, encode the rendered atlas and histology pair, and construct multiscale correlations/cost volumes.
4. Predict a residual pose update and compatibility energy from registration features while also predicting in-plane similarity and a stationary velocity field.
5. Apply the pose update, re-render the atlas plane and repeat with the same refinement weights.
6. After the final pose update, render once more and run registration again so the returned coordinate maps are bound to the final pose rather than the preceding iteration.
7. Return the best scored pose and its forward/inverse coordinate maps.

Three refinement iterations are the initial default. Iteration count is selected by an accuracy--latency ablation and may increase only if it materially improves locked development endpoints. The final ten-slice workflow must remain below the 180-second usability ceiling on the reference workstation.

## Why pose and deformation stay explicit

A free dense field can make a wrong AP or tilt superficially resemble the section. Therefore:

- plane pose remains a low-dimensional explicit state;
- residual similarity handles only in-plane nuisance geometry;
- local velocity is projected or regularized to remove bulk affine components;
- deformation magnitude and topology are bounded;
- difficult wrong-plane candidates train the compatibility/ranking objective;
- pose-only metrics are measured before any nonlinear warp.

This follows the separation of affine and non-parametric registration in Shen et al. and the use of diffeomorphic/inverse-consistent maps in VoxelMorph/GradICON, while recursive weight sharing follows recurrent registration work. The design is adapted to 2-D histology-to-3-D-atlas plane inference rather than copied wholesale.

## Pose representation

The public output is `[AP_um, LR_deg, DV_deg]`, but the trainable representation may combine:

- coarse physical bins plus continuous residuals to avoid a broad-regression local minimum;
- QuickNII-compatible O/U/V plane anchors for geometrically coherent loss;
- continuous residual corrections at every recurrent step.

Loss is evaluated in physical plane geometry as well as component coordinates. This rewards exact alignment throughout the plane, not merely approximately correct scalar angles.

## Cross-modal representation

Direct grayscale equality is inappropriate for atlas template versus real fluorescence/brightfield appearance. The model learns compatible anatomical representations using:

- shallow modality-specific grayscale/mask stems followed by a shared structural feature pyramid;
- a coarse whole-section pose distribution with continuous physical residuals;
- local multiscale correlation/cost volumes between the rendered atlas and slice;
- a genuine recurrent hidden state that receives registration evidence at every refinement;
- structural/label supervision available from exact synthetic pairs;
- aggressive grayscale appearance randomization inspired by SynthMorph's acquisition-agnostic training principle.

An intensity-only loss is never the sole registration objective.

Literal stem-weight sharing is a controlled ablation, not a premise. The
publishable coupling is joint optimization and recurrent feedback through the
shared state and structural feature space, not cosmetic reuse of every
convolution.

## Optional smart-brush conditioning

The smart-brush outline is useful side information, not anatomical truth and
not a mandatory dependency. When an outline is available, histology intensity
outside it is set to black and the outline is supplied as its own channel. An
availability channel distinguishes this assisted input from the automatic
fallback, which retains the acquired background and supplies no outline.

Training deliberately mixes accurate outlines, realistically perturbed
outlines and absent outlines. Perturbations include independent boundary
jitter, erosion/dilation, small false-positive islands and false-negative
gaps. The model-input outline remains separate from the synthetic
visible/damaged/missing-tissue masks used to gate registration losses. This
prevents a brush mistake or a real tear from being treated as deformation
ground truth. Automatic/no-user-mask and smart-brush-assisted performance are
reported as separate operating modes.

Supplying grayscale tissue and a brain mask as separate registration channels
has direct whole-brain serial-histology precedent in Lee et al. (2018), whose
objective also excludes unavailable tissue. The mixed mask/no-mask curriculum
is our robustness requirement: it preserves that useful assisted signal
without assuming an infallible segmentation or making preprocessing mandatory,
consistent with SynthMorph's broader acquisition-agnostic design principle.

## Deformation representation

The dense branch predicts a stationary velocity field on a coarse-to-fine pyramid. Scaling-and-squaring integration produces approximately diffeomorphic forward and inverse maps. Training supervises exact synthetic flow where available and adds inverse consistency, regional/boundary agreement, smoothness and Jacobian constraints. Missing or torn pixels are excluded from correspondence loss rather than hallucinated.

## Compatibility and feedback

Compatibility is a trained ranking/energy signal, not raw overlay correlation and not a confidence percentage. Each positive pair is contrasted with nearby and anatomically confusing wrong planes. At inference, low warp residual cannot excuse a low-compatibility plane. Risk--coverage and threshold-error ranking are evaluated on held-out cases; probability calibration is outside the version-1 claim.

## Multi-slice inference

The first release uses the same per-slice network plus a transparent joint optimizer over per-slice pose energies. Common tilt, partial order and user constraints are explicit latent/hard constraints. A learned set/sequence module is deferred unless an ablation shows that it improves over this interpretable solver without weakening exact constraint satisfaction.

## Matched cold-start architecture screen

Three models are trained from scratch on identical manifests, seeds, losses,
candidate lattices and view budgets. Widths are chosen before training so
parameters, MACs and peak memory are comparable.

1. **Factorized CNN control:** separate compact global-pose and pair-registration
   encoders with no recurrent hidden state.
2. **Recurrent correlation pyramid:** the leading design above, using a
   ConvGRU and local PWC/RAFT-style correlations.
3. **Windowed cross-attention pyramid:** the same outputs, recurrence and
   deformation decoder, replacing only the two coarsest correlation-fusion
   stages with local bidirectional cross-attention. It must first pass
   ONNX/DirectML feasibility.

Frozen AtlasPose, AtlasWarp and DeepSlice are comparators, never initializers.
The winning cold-start family then undergoes one-pass versus recurrent,
hidden-state versus stateless, stopped-feedback, no-ranking, raw-displacement
versus integrated-SVF, and validity-head ablations. Architecture complexity
must be earned by physical pose, correspondence, topology, error ranking and
runtime rather than by novelty.

The recurrent-correlation hypothesis is supported independently by iterative
slice-to-volume rendering in SVoRT, joint global/deformable registration in
SynthMorph and NICE-Trans, and efficient recurrent local-correlation updates
in recent medical-registration work. These precedents motivate the family;
none determines the final architecture without the matched screen.

## Evidence update: preserve topology before adding recurrence

The frozen oracle atlas-pair diagnostic showed that global mean/max
correlation summaries could not reliably rank the true local plane. Setting
source scale exactly to one did not rescue it. A subsequent parameter-matched
pair then gave one arm fixed top--bottom, left--right and diagonal 2-by-2 Haar
correlation contrasts; that arm reached only `13 / 48` fixed-candidate
decisions on each of two fresh panels and did not outperform its global
control. The defensible conclusion is narrow: these four low-order summary
families are insufficient under the tested scorer and training protocol. It
does not reject spatial correlation maps, learned aggregation or recurrence.

The next frozen mechanism test therefore retains each correlation map until a
small spatial CNN has processed it. Treatment receives the native map layout;
the matched control receives a precommitted bijective spatial permutation of
the same per-pixel correlation-and-mask vectors. Both arms have identical
active layers, parameters and initial state, and off-centre convolution weights
start at zero so their initial outputs are exact-equal. This isolates useful
anatomical neighbourhood topology without the fixed-Haar control's dormant
input columns. A recurrent state is added only if native topology first
provides reproducible ranking evidence; otherwise the cross-modal features or
candidate construction change before model complexity increases.

This sequence is consistent with mouse-histology evidence that local features
benefit from their global spatial arrangement in
[GridNet](https://doi.org/10.1093/bioinformatics/btab447), multiscale local
correlation and sequential warping in
[Dual-PRNet](https://doi.org/10.1016/j.media.2022.102379), all-pairs
correlation with recurrent updates in
[RAFT](https://doi.org/10.1007/978-3-030-58536-5_24), and iterative
slice-to-volume resampling in
[SVoRT](https://doi.org/10.1007/978-3-031-16446-0_1). These papers motivate
the components; none validates the exact proposed 2-D histology-to-3-D CCF
hybrid. The project therefore continues to require matched random-init
experiments rather than transferred weights or architectural deference to a
previous build.
