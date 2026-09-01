# Model card: unified atlas pose and dense registration

## Status

**Standalone arbitrary-plane implementation is active; untrained, unqualified and unreleased.** The deployed repository at baseline commit `c6681039e0b7acf35c9cdbee43040a3dca29cdab` contains separate AtlasPose and dense-registration candidates. The first joint implementation inherited both architectures and weights and is retained only as legacy-seeded diagnostic evidence. The new path has fresh geometry, generator, finite-thickness renderer and pose-only recurrent retrieval plumbing with no learned dependency on those systems. Its frozen image-information pilot failed the imperfect-outline consistency gate, so no learned architecture has been selected or screened. A release candidate must start randomly, cover every brain-intersecting plane, and pass later animal-split qualification, harmonized comparators, export parity and GUI validation. Measured evidence is recorded in [`docs/publication/BASELINE_LEDGER.md`](docs/publication/BASELINE_LEDGER.md).

## Intended use

The proposed model registers grayscale whole adult mouse-brain histology cut in
any brain-intersecting plane to Allen CCFv3. It jointly estimates a full 3-D
slice frame, its QuickNII O/U/V plane, in-plane scale and a nonlinear
atlas--histology coordinate map. The desktop application uses that map to
inspect alignment and translate user-marked probe observations into CCF
coordinates before trajectory and recording-site annotation.

Intended users are neuroscience researchers who review every result. The model is decision support, not an autonomous surgical, clinical or diagnostic system.

## Model design

The model family remains unselected. The current geometry candidate uses a
continuous 6-D right-handed 3-D frame, QuickNII span centre and a constrained
positive-diagonal 2-D basis for scale/anisotropy/shear, with exact O/U/V
conversion and a differentiable arbitrary-plane atlas renderer using official
`x/W,y/H` raster indices. An untrained scaffold now marginalizes declared
raster representations inside physical antipodal cells, verifies complete
catalogue coverage before normalization, predicts an initial local covariance
over two normal-tangent coordinates plus normal offset, and applies
shared-weight recurrent full-frame updates. Its exact omitted mass is
retrieval-time only; refined masses are conditional within the retrieved beam,
and all probabilities remain uncalibrated. Global pose and local
deformation remain separate so nonlinear warping cannot freely hide a wrong
atlas section; the current scaffold has no deformation decoder. No previous
trained weights initialize a release-eligible candidate. See
[`docs/publication/ARCHITECTURE.md`](docs/publication/ARCHITECTURE.md).

## Inputs

- grayscale whole-brain section from any cutting plane;
- optional automatic or smart-brush tissue outline plus an explicit availability indicator;
- Allen CCFv3 template/annotation volume rendered internally;
- optional inference constraints, handled outside model weights.

The initial public coordinate contract is documented in [`docs/publication/COORDINATE_CONVENTIONS.md`](docs/publication/COORDINATE_CONVENTIONS.md).

## Outputs

- QuickNII O/U/V and a constrained full 3-D slice frame;
- AP/L--R/D--V values only as derived legacy outputs when the coronal chart is well-conditioned;
- forward atlas-to-histology and inverse histology-to-atlas maps;
- a point pose plus implementation-stage cell masses and initial local
  covariance, all explicitly uncalibrated pending animal-held-out calibration;
- compatibility and a monotone risk score for ranking/abstention analysis;
- intermediate recurrent states and audit metadata.

## Training plan

Exact synthetic supervision comes from equal-area arbitrary Allen CCF planes
with known geometry, similarity and diffeomorphic transforms, missing-tissue
masks and diverse grayscale artifacts. Accurate, imperfect and absent outline
modes keep the smart brush optional. Curated Allen registered sections
contribute real appearance and justified pose supervision within their known
plane scope. Geodesic-normal and physical-offset candidates teach
correspondence quality to revise pose. At least three final seeds are trained,
and data scale increases toward 500k views only while preregistered validation
improves. See [`docs/publication/TRAINING_PROTOCOL.md`](docs/publication/TRAINING_PROTOCOL.md).

## Evaluation

The model must pass separate pose-only, correct-plane warp-only and end-to-end tracks; a public DeepSlice reproduction; a new hidden multi-laboratory real benchmark; an independent exact synthetic generator; dense expert landmarks/boundaries; confidence/robustness evaluation; and downstream probe phantoms. Statistical inference is animal-level. See [`docs/publication/STUDY_PROTOCOL.md`](docs/publication/STUDY_PROTOCOL.md) and [`publication/gates.yaml`](publication/gates.yaml).

## Limitations and prohibited claims

- CCF is a population reference, not an individual's complete anatomy.
- Missing or pathological tissue may lack a valid atlas correspondence.
- Synthetic accuracy cannot establish real-histology performance.
- Human consensus is uncertain and not an absolute geometric truth.
- Development compatibility/risk scores are not calibrated probabilities.
- Smart-brush-assisted results depend on user outline quality and are reported separately from fully automatic results.
- Pose credible regions may be shown as confidence only after animal-held-out
  calibration coverage passes. Propagated trajectory/region confidence also
  requires calibrated uncertainty for in-plane pose and dense mapping; a
  calibrated normal/offset distribution alone is insufficient.
- The intended model scope includes arbitrary cutting planes, but no such performance may be claimed until dedicated cardinal/extreme-oblique synthetic and animal-held-out real qualification passes; partial fields, other species and clinical use remain outside scope.
- A passed internal gate is not evidence of market-wide superiority. That language requires successful locked comparison with all applicable primary comparators.

The application must preserve manual review, landmarks and existing predictors until the joint bundle earns release approval.

## Governance

Each release carries source, data, model, preprocessing, coordinate, atlas and ontology hashes. A model that fails a hidden release gate is retained as a failed candidate, not silently tuned on that benchmark. Known failure modes and post-release amendments are versioned.
