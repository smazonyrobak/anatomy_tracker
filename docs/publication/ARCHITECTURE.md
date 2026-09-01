# Proposed unified architecture

## Design decision

The release candidate is trained from random initialization as one model with
one optimizer and one checkpoint. It has no learned dependency on AtlasPose,
AtlasWarp, DeepSlice, ImageNet weights or any earlier project checkpoint.
Those systems remain frozen comparators only. The first warm-start joint
implementation is retained as a systems prototype and development baseline;
it is ineligible for final architecture selection.

The leading hypothesis is a compact atlas-conditioned recurrent correlation
pyramid. A small random-initialized 2-D FPN produces reusable histology
features. A small random-initialized 3-D FPN produces a multiscale Allen feature
volume that is cached after training. A proposal head returns `K=4--8`
probabilistic full-plane hypotheses rather than one forced answer. Atlas feature
planes are differentiably resliced only around those hypotheses and compared
with the section through local multiscale cost volumes. A shared-weight ConvGRU
then emits a pose increment, in-plane similarity increment, compatibility
energy and matchability mask. The first one or two iterations are pose-only;
only after the pose is inside the local capture range may later iterations emit
a strongly constrained residual deformation. Updating the pose causes a fresh
atlas-feature reslice before the next iteration. Global plane pose and bounded
local deformation remain explicit even though they are optimized jointly.

Deployment uses three ONNX entry graphs exported from that same checkpoint: a
source initializer/encoder, a cached coarse candidate scorer, and a final dense
refiner. The first graph computes the source pyramid once; the scorer reuses
its coarse levels across pose candidates and recurrent updates without
constructing dense maps; the final graph decodes one pose-bound deformation.
A deterministic host-side CCF renderer runs between calls. These are entry
points into one learned model, not separately trained models. The split avoids
re-encoding the same section or integrating unused deformation fields for every
wrong-plane candidate.

The cached atlas feature pyramid is a deterministic derived asset bound to the
same checkpoint and atlas hashes and is rebuilt whenever either changes; it is
not a pretrained or separately optimized dependency.

The design is provisional until the matched cold-start architecture screen is
complete. This document defines the leading hypothesis, not a completed or
selected model.

## Arbitrary-plane implementation reuse boundary

The new arbitrary-plane model is a fresh implementation, not a widened version
of `IndependentJointModel`. That earlier network is independent of legacy
weights, but its trainable pose state, candidate renderer and losses are still
restricted to `[AP_um, LR_deg, DV_deg]` inside a coronal-family box. Its
factorized AP/LR/DV categorical head and three-coordinate recurrent update are
therefore ineligible for reuse. The associated Product-5 data path,
truth-centred candidate lattice, architecture-screen checkpoints and exported
graphs are also ineligible. AtlasPose, AtlasWarp, DeepSlice and every pretrained
or prior-project feature remain comparator-only.

Reuse is limited to verified numerical ideas and arbitrary-plane contracts:

- the torch coordinate conversions, continuous 6-D proper-frame projection,
  positive in-plane basis, QuickNII O/U/V algebra, explicit raster reflections
  and single-plane `grid_sample` primitive in `arbitrary_plane_geometry.py`;
- the v2 pose-truth RP2/offset transport and finite-frame serialization in
  `arbitrary_plane_pose_v2.py`, as supervision and audit contracts rather than
  as a learned head;
- the provenance-bound generator's finite boxcar schedule and exact normalized
  axial masses, as the reference for a new differentiable through-plane PSF
  renderer; the NumPy receipt/reduction path itself is generator-only;
- local spatial correlation, affine-free 2-D velocity projection and
  scaling-and-squaring as small independently tested numerical primitives,
  re-homed behind the new full-frame API rather than importing the old model;
- the three-channel intensity/outline/availability input semantics, with the
  v2 accurate, imperfect and absent smart-brush realizations as their source.

The forced-truth 40-proposal bank is only a premise-test and listwise-training
instrument. Its nonuniform proposal frequencies are not posterior mass and it
cannot seed inference. A release-eligible coarse head instead needs an
inference-time RP2/support-offset/roll catalogue whose base-cell masses are
defined before observing a target, followed by proposal-corrected local
residuals and top-K rerendering.

This catalogue-and-score design is an adaptation of continuous pose-density
estimation in [Implicit-PDF](https://proceedings.mlr.press/v139/murphy21a.html)
and equivolumetric hierarchical rotation cells in
[Yershova et al.](https://doi.org/10.1177/0278364909352700). Those works concern
object orientation rather than histology and do not resolve the finite-raster
or plane-offset problem here; Yershova's cells cover SO(3), while the coupled
RP2 quotient, roll and finite-frame cell measures remain this project's
derivation. They justify candidate scoring with explicit cell measure, not
copying an object-pose network or importing its weights.

The fresh implementation boundary is consequently: (1) a tensor full-frame
state and composition rule; (2) a normalized finite-thickness differentiable
atlas renderer; (3) an inference-valid probabilistic coarse retriever; (4) a
shared recurrent spatial-correlation updater for full pose; and only then (5)
an affine-free SVF decoder with explicit pose/deformation identifiability
losses. Warps select padding explicitly so black smart-brush exteriors cannot
become replicated edge intensity. For tangent or partial sections, affine
removal is evaluated over the correspondence-support domain rather than only
the whole canvas. A fresh training lineage starts in a new empty run directory;
same-lineage checkpoint resume may be enabled only after that cold start is
recorded. The existing arbitrary-plane geometry and pose contracts currently
pass their focused source tests, but this is contract evidence only. No model
family is licensed or scientifically qualified until the frozen
arbitrary-plane oracle panel has passed its live gate.

The first two numerical pieces are implemented in
`arbitrary_plane_full_frame_primitives.py` as untrained plumbing. Its physical
state is `[centre_AP_DV_ML_um(3), rotation_6d(6), log_basis_diagonal(2),
shear(1)]`; a local update right-composes SO(3), translates in the pre-update
frame and right-composes the positive in-plane basis. Raster reflection remains
a separate discrete variable. The same module renders a shared
`[channel, AP, DV, ML]` atlas through an explicit finite boxcar or Gaussian PSF
without per-pixel edge renormalization. This implementation status is not an
oracle result or permission to build or train the recurrent model early.

The through-plane renderer evaluates several differentiable samples along the
current physical normal and combines them with positive weights normalized to
unit mass. Physical thickness and the boxcar/Gaussian PSF family remain explicit
inputs rather than being absorbed into deformation. PSF-aware slice-to-volume
registration and reconstruction in
[Ebner et al.](https://doi.org/10.1007/978-3-319-52280-7_1) and
[NeSVoR](https://doi.org/10.1109/TMI.2023.3236216) motivate this acquisition
model, but both are MRI methods. The mouse-histology implementation must earn
its own convergence evidence, reported separately for smooth intensities,
label occupancies, hard categorical transitions and boundary-dominated pixels.

## Inputs and outputs

Inputs:

- grayscale histology section in the orientation selected by the user;
- optional smart-brush tissue outline in assisted mode; automatic fallback
  supplies no outline and retains the acquired background;
- an explicit outline-availability indicator, so absence of a mask is not confused with a section that fills the canvas;
- differentiably resliced Allen CCF feature/template/annotation planes at the current pose;
- recurrent state from the preceding refinement step.

Outputs:

- posterior or scored candidate representation for full plane normal, physical
  offset, in-plane orientation/localization and finite raster frame;
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
4. Predict a residual pose update and compatibility energy. Keep deformation disabled for the initial pose-only iterations; later iterations may also predict in-plane similarity and a stationary velocity field.
5. Apply the pose update, re-render the atlas plane and repeat with the same refinement weights.
6. After the final pose update, render once more and run registration again so the returned coordinate maps are bound to the final pose rather than the preceding iteration.
7. Return the best scored pose and its forward/inverse coordinate maps.

Three refinement iterations are the initial default. Iteration count is selected by an accuracy--latency ablation and may increase only if it materially improves locked development endpoints. The final ten-slice workflow must remain below the 180-second usability ceiling on the reference workstation.

## Why pose and deformation stay explicit

A free dense field can make a wrong plane orientation or offset superficially resemble the section. Therefore:

- plane pose remains a low-dimensional explicit state;
- residual similarity handles only in-plane nuisance geometry;
- local velocity is projected or regularized to remove bulk affine components;
- deformation magnitude and topology are bounded;
- difficult wrong-plane candidates train the compatibility/ranking objective;
- pose-only metrics are measured before any nonlinear warp.

This follows the separation of affine and non-parametric registration in Shen et al. and the use of diffeomorphic/inverse-consistent maps in VoxelMorph/GradICON, while recursive weight sharing follows recurrent registration work. The design is adapted to 2-D histology-to-3-D-atlas plane inference rather than copied wholesale.

[Joint SynthMorph](https://doi.org/10.1162/imag_a_00197) further supports
composing an explicit global transform with a regularized SVF while training
on synthesized label-derived image pairs with label-overlap loss. It does not
prove that the factors are identifiable in this
slice-to-volume setting. Identifiability therefore remains an empirical gate:
the deformation gauge is affine-free over valid correspondence support, pose
and deformation factors receive separate synthetic supervision, and recovery
is tested under controlled interventions and stopped-gradient ablations.

## Pose representation

The normative public output is QuickNII O/U/V plus the constrained full 3-D
frame. `[AP_um, LR_deg, DV_deg]` is a derived compatibility view only for
planes where the coronal chart is well-conditioned. The trainable
representation may combine:

- equal-area normal/physical-offset candidates plus continuous residuals to
  avoid a broad-regression local minimum;
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

Synthetic truth separates animal anatomy from slide processing. A shared
animal-level 3-D diffeomorphism maps a flat physical section in a pseudo-animal
to a mildly curved surface in atlas coordinates. Its best-fit O/U/V plane is the
global target and the residual 3-D coordinate map is deformation supervision.
A second, section-specific affine-free 2-D stationary velocity field models
mounting and handling. The network initially predicts only the 2-D in-plane
field; a very smooth normal-displacement branch is admitted only if a matched
ablation shows that it improves physical correspondence without absorbing pose
error.

Scaling-and-squaring integrates forward and inverse maps. Training supervises
the exact synthetic atlas-coordinate map and adds inverse consistency,
regional/boundary agreement, smoothness and Jacobian constraints. Affine
content is removed from the local field. Missing and torn pixels have no
correspondence; folds can be multi-valued. They remain visible for robust pose
and uncertainty training but are excluded from smooth-deformation loss.

This hierarchy follows the generative separation of shape, slice position,
contrast and non-reference signal in
[Tward et al. (2025)](https://doi.org/10.1038/s41467-025-65317-7). It replaces
the earlier simplifying assumption that all biological variation can be added
as a 2-D warp after atlas slicing.

## Compatibility and feedback

Compatibility is a trained ranking/energy signal, not raw overlay correlation
and not a confidence percentage. Each positive pair is contrasted with nearby
and anatomically confusing wrong planes. At inference, low warp residual cannot
excuse a low-compatibility plane. Risk--coverage and threshold-error ranking are
evaluated on held-out cases; calibration of this compatibility score is outside
the version-1 claim. A pose posterior is a separate output and may support
credible-volume claims only after animal-held-out calibration.

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

## Evidence update: change geometry, features and candidates before recurrence

The native-topology treatment completed the matched test but did not reproduce
a causal rescue. It reached `16 / 48` fixed-candidate decisions against `9 / 48`
for the scrambled control on fresh seed `2104322`, then tied `17 / 48` on seed
`2204322`; both were far below the frozen `46 / 48` requirement. Its
free-search physical error was also descriptively higher on both panels. Run,
pairing, topology, order, artifact and raw-result integrity all passed. The
prespecified branch is therefore to revise cross-modal features and/or
candidate construction before recurrence. This does not imply that spatial
topology or recurrence is generally unhelpful.

That diagnostic also exposed a scope mismatch with the intended product: the
v2 pose graph can describe only a coronal-family plane. The new model must
instead accept every brain-intersecting plane, including sagittal, horizontal
and extreme oblique sections. Frozen v2 generator and diagnostic sources will
remain byte-identical; arbitrary-plane support is a separate versioned path.

QuickNII O/U/V is the normative serialized and evaluation geometry because it
maps raster coordinates directly to three-dimensional CCF points for any plane,
as defined by [Puchades et al.](https://doi.org/10.1371/journal.pone.0216796).
The new discrete renderer follows QuickNII/webnutil pixel indexing exactly:
pixel `(x,y)` in a `W`-by-`H` raster maps to `O+(x/W)U+(y/H)V`, without a
half-pixel term. The frozen earlier `+0.5` plane-distance diagnostic is retained
only for historical reproducibility and cannot define future interoperability
or release evaluation.
The learned internal state is constrained rather than nine unconstrained
coordinates: a 3-D QuickNII span centre, a continuous 6-D representation of a
right-handed orthonormal frame `[u,v,n]`, and a positive-diagonal
upper-triangular 2-by-2 in-plane basis `A`. Exact conversion uses
`[U V] = [u v] A` and `O = centre - (U+V)/2`. The two log-diagonal terms and
one shear term are necessary to round-trip general O/U/V and the historical
double-tilt raster exactly; forcing one scale would silently move shear into the
dense field. Recurrent refinement, when later justified, composes a small
`so(3)` rotation, local-frame 3-D translation and bounded in-plane-basis
increment. The dense SVF remains affine-free. Raster flips are explicit O/U/V
reparameterizations, never learned physical reflections. A discrete horizontal
flip uses `O'=O+((W-1)/W)U, U'=-U`; its vertical counterpart uses `H` in the
same way. Raster dimensions are therefore part of every flip contract.

The continuous rotation choice follows the topological and empirical argument
of [Zhou et al.](https://doi.org/10.1109/CVPR.2019.00589): Euler and ordinary
quaternion coordinates have unavoidable Euclidean discontinuities for global
regression, whereas two 3-D vectors projected by Gram--Schmidt give a
continuous 6-D representation. Quaternions remain possible inside a later
proper directional distribution, but are not used as an unqualified global
point-regression target.

The reference synthetic orientation distribution samples normals by the
normalized-Gaussian sphere method of
[Muller](https://doi.org/10.1145/377939.377946), quotients antipodes, samples
roll uniformly, and samples the coupled signed normal/plane offset
uniformly over the annotation support projected onto that normal. Rendering
then verifies nonempty support. Training may add named rare/tangent stress
strata, but they cannot silently change the reference validation measure.
Tiny-support or symmetry-ambiguous slices remain in uncertainty/failure
training; point-accuracy summaries use a frozen visible-support threshold and
report abstention rather than pretending an unidentifiable plane is exact.

The initial probabilistic geometry design uses equal-area antipodal normal
cells, conditional physical-offset candidates and local tangent-space
covariance for two normal rotations plus normal offset. It preserves
multimodality without forcing independent Gaussian Euler angles. Point
estimates remain explicit, and the probabilistic head is retained only if it
does not reduce their accuracy. On unseen animals, credible spatial volumes
will be checked for nominal coverage before their uncertainty can propagate to
electrode trajectories or region assignments.

This initial covariance is only a three-degree-of-freedom plane posterior. It
does not represent uncertainty in the complete finite in-plane frame/basis or
the dense deformation, so it is insufficient for downstream trajectory volumes
or region-assignment confidence. Those outputs remain explicitly uncalibrated
and unavailable for confidence propagation until full finite-frame and
deformation uncertainty are both represented and animal-held-out calibrated.

Each of the initial `K=4--8` components carries a finite-frame/O/U/V mean, a
positive-definite local tangent-space covariance, mixture mass and a discrete
probability over equivalent raster reparameterizations, never an anatomical
reflection. Training marginalizes the exact equivalent raster
representations rather than selecting an arbitrary normal sign or encoding an
improper reflection as a rotation. Physical O/U/V anchor loss remains alongside
mixture likelihood and candidate-ranking loss so uncertainty cannot improve by
sacrificing the point estimate. A small independent deep ensemble is considered
only after the single-model posterior is stable; calibration is fitted later on
held-out animals.

The antipodal representation applies to an unoriented infinite plane:
`(n,d)` and `(-n,-d)` are one object, so normal and signed offset are never
canonicalized independently. The finite raster frame additionally retains its
in-plane basis and explicit reflection state. A later fully continuous
alternative may use a [Bingham distribution](https://doi.org/10.1214/AOS/1176342874).
If the first probabilistic candidate is adopted, it will use proposal-corrected
candidate mass plus local tangent-space residuals to preserve separated modes
without special-function normalizers; the current forced-centre candidate set
is a proposal, not posterior mass.

Implementation proceeds in this order: geometry conversions and a general
differentiable renderer; a provenance-bound manifest and generator;
geodesic-normal/physical-offset/roll candidates with finite-frame and in-plane
localization; the new data and pose-truth contract; then a small arbitrary-plane
oracle ranking premise test. The arbitrary-plane generator v2 source path now
covers subject deformation, finite thickness, section
processing, appearance, damage, accurate/imperfect/absent smart-brush modes,
final provenance-bound realizations, finite wrong-plane candidates and an
independently verifiable oracle-panel runner. This establishes source and unit
contracts, not a scientific result: the frozen live panel and its null gate
must still complete before recurrence is built. Existing Product-5 data are
historical coronal audit evidence only and cannot enter release training,
validation, features or pseudolabels. DeepSlice Ground Truth (DOI
`10.25949/22802411`) remains benchmark-only. Neither validates arbitrary-plane
performance; external real validation must deliberately span cardinal and
extreme-oblique cutting planes and remain split by animal.
