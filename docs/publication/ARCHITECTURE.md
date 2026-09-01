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
probabilistic full 3-D plane-frame/O/U/V pose. The current CCF plane is rendered, encoded in
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

- posterior or scored candidate representation for full plane normal, physical
  offset and finite raster frame;
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

A free dense field can make a wrong plane orientation or offset superficially resemble the section. Therefore:

- plane pose remains a low-dimensional explicit state;
- residual similarity handles only in-plane nuisance geometry;
- local velocity is projected or regularized to remove bulk affine components;
- deformation magnitude and topology are bounded;
- difficult wrong-plane candidates train the compatibility/ranking objective;
- pose-only metrics are measured before any nonlinear warp.

This follows the separation of affine and non-parametric registration in Shen et al. and the use of diffeomorphic/inverse-consistent maps in VoxelMorph/GradICON, while recursive weight sharing follows recurrent registration work. The design is adapted to 2-D histology-to-3-D-atlas plane inference rather than copied wholesale.

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
differentiable renderer; a provenance-bound v3 manifest and generator;
geodesic-normal/physical-offset candidates; the new data and posterior
contract; then a small arbitrary-plane oracle ranking premise test. The finite
CCF render precursor has passed its pinned-Allen development preflight, while
complete deformation/appearance/outline realization and finite wrong-plane
candidates remain next. Recurrence is built only after the oracle premise is
adequate. Existing Product-5 and DeepSlice data remain useful coronal
auxiliary/comparator evidence, but cannot validate arbitrary-plane
performance; external real validation must deliberately span cardinal and
extreme-oblique cutting planes and remain split by animal.

## Evidence update: imperfect-brush gate failure

The complete 64-case model-free image-information pilot passed 30 of 31 atomic
checks but is `FAIL`. Context MIND top-1 was `63/64` without an outline,
`62/64` with an accurate outline, and `52/64` with an imperfect outline. The
11-case imperfect-outline deficit exceeded the frozen maximum of six.

Descriptive support-aware scoring recovered all 11 paired losses, making
candidate-support handling the leading follow-up mechanism, but privileged
support cannot become a model input or retroactively replace the primary gate.
Consequently the image encoder must treat outline presence explicitly, retain
raw-background input as a complete path, and avoid interpreting black exterior
as anatomical evidence. No learned recurrent screen is authorized by this
result. Implementation of the probabilistic retriever, renderer, and shared
updater may continue, while any scientific screen waits for a separately frozen
paired mask-mechanism test with unchanged shuffled controls.
