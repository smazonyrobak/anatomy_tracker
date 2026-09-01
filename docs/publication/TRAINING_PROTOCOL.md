# Unified-model training protocol

## Objective

Train one pose-and-registration network end to end for every brain-intersecting
section plane, including coronal, sagittal, horizontal and extreme oblique
cuts. The model must learn exact global 3-D slice-frame geometry, bounded local
correspondence and wrong-plane rejection. Its geometry output includes a point
estimate and an initial tractable probabilistic posterior over plane normal and
physical offset. Its local covariance has only three degrees of freedom--two
normal rotations and one normal offset--so it is not a full finite-frame or
deformation posterior and cannot support trajectory or region-assignment
confidence. Full in-plane-frame and deformation uncertainty must be represented
and calibrated on held-out animals first; the probabilistic design is retained
only if it does not reduce point-estimate accuracy. Compatibility remains a
monotone risk score for ranking and abstention unless and until it separately
earns probability calibration. Model
scale, posterior complexity and iteration count are selected on development
data under the runtime budget; no architecture is promoted merely because it
is newer or larger.

## Stage 0: development safeguards

During generator and early architecture development:

1. preserve source provenance and nullable synthetic or exact real
   animal/specimen/experiment identifiers in every manifest;
2. use only separately salted training and development manifests; do not mint
   or inspect a final-test split for the new arbitrary-plane method yet;
3. version coordinate, preprocessing, mask and output contracts used by each
   premise test;
4. capture the baseline software at `c6681039e0b7acf35c9cdbee43040a3dca29cdab`;
5. record environment, GPU and dependency provenance, with an explicitly
   empty learned-checkpoint dependency list for every release-eligible run.

After the method stabilizes, animal-level development/final-test manifests,
benchmark custody, primary endpoints, comparator revisions and release gates
are frozen before the affected benchmark is opened, as specified under Future
validation below.

## Stage 1: generator and premise tests

Use a small audit set, not a performance claim. First validate the structured
3-D frame, exact QuickNII O/U/V conversion, cardinal and arbitrary-plane
rendering, official `x/W,y/H` raster indexing, dimension-aware discrete
raster-flip reparameterizations, artifacts, masks and topology. The frozen
legacy `+0.5` diagnostic is not an eligible interoperability fixture.
Sample the reference plane distribution with equal-area antipodal normals,
uniform in-plane roll and uniform valid annotation-support offset; keep named
tiny-support and tangent-plane stress strata separate. Perturb normals by
geodesic angle and offsets in physical distance, then test whether candidate
features rank the true plane. Failure prevents recurrence and prevents
describing warp residual as pose confidence.

## Stage 2: 20k--50k controlled experiments

Train under matched views, seeds and compute:

- a from-scratch factorized CNN pose/registration control;
- a from-scratch recurrent correlation-pyramid model;
- a from-scratch windowed cross-attention pyramid after export preflight;
- the selected family with one, four and eight shared-weight refinement steps;
- the selected family with registration-to-pose gradient stopped;
- the selected family without wrong-plane ranking;
- the selected family with bounded integrated SVF versus raw displacement.

Frozen AtlasPose followed by AtlasWarp is evaluated as a historical comparator
but cannot enter training or initialize any release-eligible candidate.

The development rule selects the smallest model within the prespecified practical-equivalence margin of the best candidate. Three iterations are the initial deployment hypothesis; more iterations require measured accuracy gain and remain within runtime limits.

## Stage 3: 100k multi-seed qualification

Train the selected family with at least three independent seeds. Report each seed, mean and variability. Early stopping uses the frozen joint validation endpoints, not training loss alone. A candidate failing numerical, topology or risk-ranking requirements is ineligible even if its mean overlap is high.

## Stage 4: scale only with evidence

Increase to 200k and up to 500k unique synthetic views only if learning and validation curves show credible improvement without widening the real-domain gap. Data volume is a ceiling, not a target. The final choice is made before hidden-test access.

## Curriculum

1. **Foundational exact supervision:** train every randomly initialized branch
   on exact/near-exact atlas frames across the full arbitrary-plane reference
   distribution, mild deformation, supervised O/U/V geometry and flow.
2. **Scheduled closed loop:** a growing fraction of model-predicted poses and nearby hard negatives.
3. **Full closed loop:** recurrent predictions, the full Allen/CCF-derived
   synthetic artifact distribution and difficult anatomical negatives. The
   earlier coronal joint screen mixed Product-5 real sections with synthetic
   batches; that regimen is retained as a historical legacy record and is
   release-ineligible for the standalone arbitrary-plane model.

No prior project or external vision weights are used by the primary release
screen. Legacy-seeded runs remain clearly labelled exploratory diagnostics and
cannot be promoted. A later transfer-learning sensitivity analysis may be
reported separately, but it cannot determine the shipped weights.

## Loss families

- physical O/U/V plane-anchor loss, structured frame loss and local
  tangent-space pose likelihood;
- coarse-bin/residual loss where used;
- singleton positive-versus-hard-negative listwise ranking on exact synthetic
  arbitrary-plane pairs; the earlier Product-5 pair loss is a historical legacy
  diagnostic and is release-ineligible for the standalone model;
- sign-balanced geodesic-normal and physical-offset negatives for arbitrary
  synthetic planes; the legacy six one-axis Product-5 negatives are retained
  only as historical coronal audit evidence and cannot train or select the
  standalone release weights;
- forward and inverse visible-pixel endpoint error on exact synthetic pairs;
- label/hierarchy and boundary correspondence;
- inverse and gradient-inverse consistency;
- velocity smoothness and Jacobian/topology terms;
- deep supervision at every recurrent state;
- real-domain consistency/pose supervision only in separately labelled
  historical or diagnostic runs that are release-ineligible for the standalone
  arbitrary-plane model.

Loss weights are chosen on development data within a logged search budget. They are not altered after hidden-test access. No intensity similarity is allowed to dominate cross-modal anatomical supervision.

## Optimization and monitoring

Every run records:

- full resolved configuration and git commit;
- all source/data/manifest hashes and the empty learned-checkpoint dependency receipt;
- seed, optimizer, learning-rate schedule, batch size and precision;
- training/validation curves by component and artifact stratum;
- throughput, wall time, GPU utilization, peak memory and checkpoint times;
- best/last checkpoints and exact early-stopping reason;
- non-finite gradients, rejected batches and process interruptions.

Complete rendered synthetic manifests record the exact normal, in-plane axes,
centre, O/U/V, roll, offset quantile, tissue support, RNG stream and rejection
attempt, CCF asset identity, and nullable animal/specimen/experiment fields.
The v2 source chain now binds the positive plane frame, finite subject-deformed
slab, section processing, intended and effective O/U/V, exact coordinate maps,
CCF intensity/annotation/mask, appearance, background, damage and
accurate/imperfect/absent outline descendants into layered IDs and a final
`synthetic_realization_id`. This is source-contract evidence only: the retired
two-resolution universal slab gate remains an authenticated rejection, and a
fixed-case four-resolution replacement must pass before large-scale training.
Real records retain their exact animal, specimen and experiment IDs.
Smart-brush output remains helpful optional evidence, never a required input
or anatomical-loss gate.

Training runs detached and resumable on the designated fast workspace. A progress JSON and plain terminal log update throughout. Monitoring is periodic and must not interrupt a healthy process. Checkpoints are written atomically; a short CPU run verifies exact model/optimizer/scheduler/RNG/data-stream continuation. CUDA resume restores the same state, but bitwise equivalence is not claimed because `grid_sample` backward can be nondeterministic on CUDA.

## Candidate selection

There is no undisclosed composite score. Candidate eligibility first requires all safety/topology, absolute accuracy and runtime gates. Among eligible candidates, the prespecified ordering is:

1. real validation landmark TRE, the unique anatomical primary endpoint;
2. secondary pose-track plane distance;
3. exact synthetic correspondence and tail error;
4. risk--coverage and error-ranking performance;
5. runtime and model size, choosing the simpler candidate inside the practical-equivalence margin.

## Export

The canonical release loader selects `ema.shadow` from the validation-selected PyTorch checkpoint and records the checkpoint hash and state selector in an export receipt. That one selected state is exported through three ONNX entry graphs: a source initializer/encoder, a cached coarse candidate scorer and a final dense refiner. The deterministic host atlas renderer connects them between refinement iterations. CPU and DirectML outputs are compared with frozen tolerances on normal, severe, boundary and flip cases, using dynamic batches of one and ten. The bundle includes model metadata, preprocessing and coordinate contract versions, source/checkpoint hashes, supported providers and known limitations. Export parity is necessary but is not evidence of anatomical validity.

## Hidden qualification

The final bundle is evaluated once on the locked benchmarks. A failing result remains recorded; training cannot silently continue against that test. Further development creates a new model version and requires a newly protected confirmatory set for any renewed final claim.

Full benchmarking waits until the method stabilizes. At that point splits are
strictly by animal with untouched final-test animals; Allen/synthetic data are
training sources, DeepSlice Ground Truth (DOI `10.25949/22802411`) is a public
benchmark, and separate lab histology supplies external validation where
available. Blinded expert alignments define the reference. The primary endpoint
is physical landmark registration error. Corresponding-plane distance and
angle are secondary pose-track metrics; regional overlap, failures, correction
time, effect sizes and 95% confidence intervals are also reported using animals
as statistical units. Exact
splits, code, configurations, seeds and raw predictions are retained. Nominal
credible-set coverage is assessed on unseen animals (for example, whether 90%
regions contain the reference about 90% of the time). The current three-DOF
normal/offset covariance is insufficient for downstream confidence: trajectory
volumes and brain-region probabilities remain uncalibrated and unused until
full finite-frame and deformation uncertainty are represented and calibrated.
Repeated slices
are treated hierarchically rather than as exchangeable calibration cases,
consistent with the grouped prediction-set setting of
[Dunn et al.](https://doi.org/10.1080/01621459.2022.2060112).
