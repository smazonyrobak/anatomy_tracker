# Joint constraints and electrode uncertainty: implementation decision

2026-09-28. This is a literature-informed implementation proposal, not a model
accuracy or calibration result. It updates the intended behavior beyond the
deterministic candidate filtering described in `CONSTRAINTS.md`.

## Keep one joint model

Keep the randomly initialized v6 recurrent model and one trained checkpoint.
Its deformation already changes the atlas comparison used by the next pose
update (`absolute_deformation_warps_next_finite_thickness_render` in
`arbitrary_plane_joint_model_v6.py`). Preserve gradients through that feedback.
A pose-first training curriculum is compatible with a joint final model; a
permanently frozen alignment network feeding a separate warp network is not
the intended deliverable.

[SVoRT](https://arxiv.org/abs/2206.10802) supports iterative slice-transform
updates using current reconstruction and other slices. [Joint
SynthMorph](https://arxiv.org/html/2301.11329v3) supports explicit global and
deformable components with broad synthetic appearance variation. Neither
establishes arbitrary-plane mouse-histology accuracy. The finite-thickness
renderer remains motivated by [NeSVoR](https://doi.org/10.1109/TMI.2023.3236216);
its image-noise variances are not calibrated electrode-coordinate uncertainty.

Keep global pose explicit, project bulk affine motion out of the residual
velocity over valid tissue, and penalize excessive deformation. Supervise
physical coordinate correspondence and pose separately where synthetic truth
exists. A visually good warp of the wrong atlas plane is a failure. Missing
tissue remains missing correspondence, not a deformation target.

## Integrate constraints into inference and training

Use the current image-derived multimodal distribution as a proposal. Introduce
explicit measurement factors for electrode observations and surgery, and let
their residuals influence every recurrent update. In schematic notation,

`q(z, trajectory | images, marks, constraints) ∝ qθ(z | images) ×
p(marks | z, trajectory) × p(trajectory | surgery) × p(stack geometry) × 1_hard`.

Here `z` contains every section's full frame and deformation. This is an
approximate conditional model; the factorization assumptions must be checked.
Do not multiply the same surgical evidence twice if it already enters the
learned proposal distribution. Start with an image-only proposal plus explicit
factors, then amortize the conditioned update using synthetic constraint
realizations.

- Rough entry position: a physical-coordinate likelihood with declared
  covariance. An entry radius must explicitly mean a hard disk or a specified
  coverage radius; it must not silently become one standard deviation.
- Rough angle: a directional likelihood around the insertion direction,
  expressed in a sphere tangent space or a von Mises--Fisher distribution.
  Probe direction from entry to tip is directed; the slice-plane normal is
  antipodal. They are different geometric variables.
- Slice range/order/common cutting direction: factors in full physical plane
  geometry. A coronal AP interval is not a general sagittal/oblique range.
- Explicit hard bounds: zero probability outside the feasible set, maintained
  during refinement and sampling. Learned penalties alone do not guarantee
  satisfaction. Report infeasibility instead of silently relaxing a bound.

Map annotated 2-D electrode points through the current pose and warp, fit or
update the shared physical trajectory, and feed standardized point-to-track,
entry and angle residuals plus availability indicators into the recurrent
state. Backpropagate their likelihood through pose and deformation. Entry and
angle information cannot locate an otherwise unobserved track without some
link to marked/detected electrode evidence. No electrode marks means those
track-to-image factors are absent.

Condition coarse candidates before pruning, and recompute the factors after
each pose/warp update. Retain probability outside refined candidates; expand
the candidate set when conditioning exposes an unrendered mode. A tiny feasible
fragment of an image-incompatible posterior is a conflict signal, not evidence
of a certain alignment. Preserve an unconstrained result for comparison.

Generate noisy, missing, incorrect-but-feasible and inconsistent constraints
from known synthetic trajectories. Randomly omit constraints during training.
First experiment: paired identical animals with constraints absent, realistic,
and deliberately wrong; compare physical correspondence error, trajectory/site
error, failures and runtime. Ablate both deformation feedback and constraint
feedback to establish that joint information actually improves predictions.

## Uncertainty required for an electrode-site probability

The existing three-coordinate plane covariance is insufficient. Keep global
mode probabilities and add a compact correlated local distribution over the
full nine physical frame degrees of freedom and smooth deformation. A
practical first extension is a shared low-dimensional Gaussian latent driving
both frame residuals and low-resolution affine-free velocity perturbations.
This preserves pose--warp correlation and spatially coherent samples; its rank
is an internal development choice. [Probabilistic
VoxelMorph](https://doi.org/10.1016/j.media.2019.07.006) motivates sampling
deformations via a probabilistic velocity model, not a claim that such samples
are automatically calibrated here.

Sample the joint frame/warp, uncertain electrode annotations, entry/depth/tip
reference, and known probe geometry; solve the constrained trajectory for each
sample. Shared cutting orientation, coordinate-reference errors and animal
variation must remain shared across that animal's sections and sites. Treating
all section errors as independent would make confidence spuriously improve
with more slices. Unknown atlas-to-individual boundary variation must be
represented or clearly excluded from the reported probability's scope.

For site j, estimate the region probability by the weighted fraction of
physical site samples falling in that atlas region. Preserve omitted candidate
mass as unresolved probability; do not normalize retained candidates to a
false 100%. Report ontology/version and assumptions. An SD ellipsoid is useful
for a single roughly Gaussian mode; multimodal solutions need sample clouds or
possibly disconnected credible volumes. Distinguish a pointwise 90% site region
from a simultaneous 90% tube containing the entire trajectory.

Fit calibration only on separate calibration animals and evaluate the complete
conditioned pipeline on unseen animals, including constraint quality strata.
Temperature scaling is a small initial categorical calibration method
([Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)); also assess spatial
coverage, width, NLL/Brier scores and anatomical-site reliability. Preserve
point accuracy. Sample-based conformal sets can supplement the spatial output
([Wang et al.](https://proceedings.mlr.press/v206/wang23n.html)), but their
marginal coverage guarantee does not establish a 90% posterior probability for
one particular site's TRN membership. Calibration must cover shared-animal
dependence and actual deployment assumptions.

## Benchmark and delivery gate

After promising internal animal-level results, freeze the method and compare
the automatic mode with [DeepSlice](https://www.nature.com/articles/s41467-023-41645-4)
on the shared coronal domain. Keep smart-brush/surgical-assisted comparisons
separate and match supplied information. Report arbitrary-plane capability
separately; unsupported DeepSlice orientations are not failures in the primary
paired superiority analysis. Pose-only and final anatomical-warp endpoints are
different and both matter.

The existing validation plan records previous exposure to related public
DeepSlice cohorts. The public benchmark is therefore a transparent frozen
comparison, not an untouched final test. Final superiority needs independent
real animals, blinded anatomical references, paired animal-level effects with
95% intervals, and retained failures. A disappointing public result cannot be
repeatedly tuned away while still calling that dataset final-test data.

The next implementation should extend the existing joint model and inference
functions, not add an application or a new framework. All development assets
stay on I:. Ship one trained model usable by the existing Anatomy Tracker,
with full pose/warp and optional constraints; enable numerical confidence
claims only when the end-to-end calibration evidence supports them.
