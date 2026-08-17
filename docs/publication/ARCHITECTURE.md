# Proposed unified architecture

## Design decision

The proposed system is one jointly trained model with one optimizer and one checkpoint. It is not two independently optimized models joined by a heuristic vote. The pose initializer and paired-image registrar retain geometry-appropriate encoders because their input contracts differ: the initializer consumes a canonicalized `3 x 299 x 299` view, while registration consumes metrically meaningful `2 x 320 x 464` atlas/histology pairs. One pose-review core is reused at every render--compare--correct iteration, and gradients couple pose correction, compatibility and dense registration during training. Explicit output heads retain the distinction between global atlas-plane pose and local in-plane deformation.

Deployment uses two ONNX entry graphs exported from that same checkpoint: an initializer graph and a recurrent-refiner graph. A deterministic host-side CCF renderer runs between them. This is an implementation boundary forced by the large 3-D atlas and current ONNX/DirectML volumetric-sampling constraints, not two separately trained models.

The design is provisional until the preregistered development ablations are complete. This document defines the initial hypothesis, not a completed implementation.

## Inputs and outputs

Inputs:

- grayscale histology section in the orientation selected by the user;
- visible-tissue mask, obtained automatically in automatic mode or supplied by the smart brush in assisted mode;
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

- the proven ConvNeXt pose encoder for canonical whole-section localization;
- the proven tied Siamese registration encoder for metrically preserved atlas/histology pairs;
- a shared recurrent review head that sees the fixed plane, warped moving slice, masks, structural residuals, current pose and registration summaries;
- multiscale correlation/cost volumes;
- structural/label supervision available from exact synthetic pairs;
- aggressive grayscale appearance randomization inspired by SynthMorph's acquisition-agnostic training principle.

An intensity-only loss is never the sole registration objective.

Literal encoder-weight sharing is a controlled ablation, not a premise. Forcing unlike geometric inputs through one encoder would discard useful pretrained representations and could weaken both tasks. The publishable coupling is joint optimization and recurrent feedback through the shared review state, not cosmetic reuse of every convolution.

## Deformation representation

The dense branch predicts a stationary velocity field on a coarse-to-fine pyramid. Scaling-and-squaring integration produces approximately diffeomorphic forward and inverse maps. Training supervises exact synthetic flow where available and adds inverse consistency, regional/boundary agreement, smoothness and Jacobian constraints. Missing or torn pixels are excluded from correspondence loss rather than hallucinated.

## Compatibility and feedback

Compatibility is a trained ranking/energy signal, not raw overlay correlation and not a confidence percentage. Each positive pair is contrasted with nearby and anatomically confusing wrong planes. At inference, low warp residual cannot excuse a low-compatibility plane. Risk--coverage and threshold-error ranking are evaluated on held-out cases; probability calibration is outside the version-1 claim.

## Multi-slice inference

The first release uses the same per-slice network plus a transparent joint optimizer over per-slice pose energies. Common tilt, partial order and user constraints are explicit latent/hard constraints. A learned set/sequence module is deferred unless an ablation shows that it improves over this interpretable solver without weakening exact constraint satisfaction.

## Alternatives retained for ablation

- frozen AtlasPose followed by frozen AtlasWarp;
- single-pass joint model without recurrence;
- recurrent model with registration-to-pose gradient stopped;
- recurrent model without wrong-plane ranking;
- unrestricted dense field;
- pose-only and correct-plane warp-only variants.

The selected architecture must earn its complexity against these controls.
