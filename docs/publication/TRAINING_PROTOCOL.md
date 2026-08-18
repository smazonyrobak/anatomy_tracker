# Unified-model training protocol

## Objective

Train one recurrent pose-and-registration network end to end. The model must learn exact global plane geometry, bounded local correspondence and wrong-plane rejection. Its compatibility head supplies a monotone risk score for ranking and abstention analysis; version 1 does not claim that this score is a calibrated probability. Model scale and iteration count are selected on development data under the runtime budget; no architecture is promoted merely because it is newer or larger.

## Stage 0: freeze the experiment

Before model code is trained:

1. create specimen-level train/validation/test manifests and hashes;
2. assign benchmark custody and lock the new hidden real test;
3. freeze coordinate, preprocessing, mask and output contracts;
4. freeze primary endpoints, statistical plan, comparator revisions and release gates;
5. capture the baseline software at `c6681039e0b7acf35c9cdbee43040a3dca29cdab`;
6. record environment, GPU and dependency provenance, with an explicitly
   empty learned-checkpoint dependency list for every release-eligible run.

## Stage 1: generator and premise tests

Use a small audit set, not a performance claim. Validate exact geometry, artifacts, masks and topology. Perturb known poses and test whether existing/new registration features rank the true plane above neighbouring candidates. Failure of this test prevents describing warp residual as pose confidence.

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
   on exact/near-exact atlas planes, mild deformation, supervised pose and flow.
2. **Scheduled closed loop:** a growing fraction of model-predicted poses and nearby hard negatives.
3. **Full closed loop:** recurrent predictions, full artifact distribution, difficult anatomical negatives and mixed real/synthetic batches.

No prior project or external vision weights are used by the primary release
screen. Legacy-seeded runs remain clearly labelled exploratory diagnostics and
cannot be promoted. A later transfer-learning sensitivity analysis may be
reported separately, but it cannot determine the shipped weights.

## Loss families

- physical plane-anchor loss and component pose likelihood;
- coarse-bin/residual loss where used;
- singleton positive-versus-hard-negative compatibility ranking on exact
  synthetic pairs;
- Product-5 acceptable-set listwise cross-entropy over the frozen
  sub-resolution set (AP 25/50 um and L--R/D--V 0.25/0.5 degrees), with
  balanced signed resolvable negatives in every real list;
- forward and inverse visible-pixel endpoint error on exact synthetic pairs;
- label/hierarchy and boundary correspondence;
- inverse and gradient-inverse consistency;
- velocity smoothness and Jacobian/topology terms;
- deep supervision at every recurrent state;
- real-domain consistency/pose supervision only where its labels justify it.

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

Training runs detached and resumable on the designated fast workspace. A progress JSON and plain terminal log update throughout. Monitoring is periodic and must not interrupt a healthy process. Checkpoints are written atomically; a short CPU run verifies exact model/optimizer/scheduler/RNG/data-stream continuation. CUDA resume restores the same state, but bitwise equivalence is not claimed because `grid_sample` backward can be nondeterministic on CUDA.

## Candidate selection

There is no undisclosed composite score. Candidate eligibility first requires all safety/topology, absolute accuracy and runtime gates. Among eligible candidates, the prespecified ordering is:

1. hidden-from-training real validation plane distance;
2. real validation landmark TRE;
3. exact synthetic correspondence and tail error;
4. risk--coverage and error-ranking performance;
5. runtime and model size, choosing the simpler candidate inside the practical-equivalence margin.

## Export

The canonical release loader selects `ema.shadow` from the validation-selected PyTorch checkpoint and records the checkpoint hash and state selector in an export receipt. That one selected state is exported as an initializer ONNX graph and a recurrent-refiner ONNX graph; the deterministic host atlas renderer connects them between refinement iterations. CPU and DirectML outputs are compared with frozen tolerances on normal, severe, boundary and flip cases, using dynamic batches of one and ten. The bundle includes model metadata, preprocessing and coordinate contract versions, source/checkpoint hashes, supported providers and known limitations. Export parity is necessary but is not evidence of anatomical validity.

## Hidden qualification

The final bundle is evaluated once on the locked benchmarks. A failing result remains recorded; training cannot silently continue against that test. Further development creates a new model version and requires a newly protected confirmatory set for any renewed final claim.
