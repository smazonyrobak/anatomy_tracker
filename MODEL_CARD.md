# Model card: unified atlas pose and dense registration

## Status

**Research scaffold complete; independent architecture selection is active; not qualified or released.** The deployed repository at baseline commit `c6681039e0b7acf35c9cdbee43040a3dca29cdab` contains separate AtlasPose and dense-registration candidates. The first joint implementation inherited both architectures and weights and is retained only as legacy-seeded diagnostic evidence. The release candidate will be trained from random initialization and remains unselected until matched architecture screening, multiseed qualification, harmonized comparators, export parity and GUI validation pass. Measured evidence is recorded in [`docs/publication/BASELINE_LEDGER.md`](docs/publication/BASELINE_LEDGER.md).

## Intended use

The proposed model registers grayscale whole coronal adult mouse-brain histology to Allen CCFv3. It jointly estimates bregma-centred AP position, L--R/D--V cutting tilt, in-plane similarity and a nonlinear atlas--histology coordinate map. The desktop application uses that map to inspect alignment and translate user-marked probe observations into CCF coordinates before trajectory and recording-site annotation.

Intended users are neuroscience researchers who review every result. The model is decision support, not an autonomous surgical, clinical or diagnostic system.

## Model design

The leading candidate is one randomly initialized recurrent correlation-pyramid system stored in one PyTorch checkpoint. A coarse probabilistic pose head, differentiable atlas renderer, multiscale structural correlations and a shared ConvGRU state jointly refine global plane pose, in-plane similarity and an integrated stationary velocity field. Global pose and local deformation remain separate so nonlinear warping cannot freely hide a wrong atlas section. A factorized CNN and a windowed cross-attention pyramid are matched cold-start alternatives; no family is selected before the prespecified screen. Deployment will export entry graphs from the same selected checkpoint, connected by the deterministic atlas renderer. See [`docs/publication/ARCHITECTURE.md`](docs/publication/ARCHITECTURE.md).

## Inputs

- grayscale whole coronal section;
- visible-tissue mask, automatic or user assisted;
- Allen CCFv3 template/annotation volume rendered internally;
- optional inference constraints, handled outside model weights.

The initial public coordinate contract is documented in [`docs/publication/COORDINATE_CONVENTIONS.md`](docs/publication/COORDINATE_CONVENTIONS.md).

## Outputs

- AP in µm from bregma, anterior positive;
- L--R and D--V tilt in degrees;
- forward atlas-to-histology and inverse histology-to-atlas maps;
- a point pose plus candidate-posterior/local-covariance outputs for later
  animal-held-out calibration;
- compatibility and a monotone risk score for ranking/abstention analysis;
- intermediate recurrent states and audit metadata.

## Training plan

Exact synthetic supervision comes from oblique Allen CCF planes with known similarity and diffeomorphic transforms, missing-tissue masks and diverse grayscale artifacts. Curated Allen registered sections contribute real appearance and justified pose supervision. Wrong-plane candidates teach correspondence quality to revise pose. At least three final seeds are trained, and data scale increases toward 500k views only while preregistered validation improves. See [`docs/publication/TRAINING_PROTOCOL.md`](docs/publication/TRAINING_PROTOCOL.md).

## Evaluation

The model must pass separate pose-only, correct-plane warp-only and end-to-end tracks; a public DeepSlice reproduction; a new hidden multi-laboratory real benchmark; an independent exact synthetic generator; dense expert landmarks/boundaries; confidence/robustness evaluation; and downstream probe phantoms. Statistical inference is animal-level. See [`docs/publication/STUDY_PROTOCOL.md`](docs/publication/STUDY_PROTOCOL.md) and [`publication/gates.yaml`](publication/gates.yaml).

## Limitations and prohibited claims

- CCF is a population reference, not an individual's complete anatomy.
- Missing or pathological tissue may lack a valid atlas correspondence.
- Synthetic accuracy cannot establish real-histology performance.
- Human consensus is uncertain and not an absolute geometric truth.
- Development compatibility/risk scores are not calibrated probabilities.
- Pose credible regions and downstream trajectory/region probabilities may be
  shown as confidence only after animal-held-out calibration coverage passes.
- The model is scoped to whole coronal adult mouse-brain sections and must not be assumed valid for partial fields, other species, sagittal/horizontal sections or clinical decisions.
- A passed internal gate is not evidence of market-wide superiority. That language requires successful locked comparison with all applicable primary comparators.

The application must preserve manual review, landmarks and existing predictors until the joint bundle earns release approval.

## Governance

Each release carries source, data, model, preprocessing, coordinate, atlas and ontology hashes. A model that fails a hidden release gate is retained as a failed candidate, not silently tuned on that benchmark. Known failure modes and post-release amendments are versioned.
