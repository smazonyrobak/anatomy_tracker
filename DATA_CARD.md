# Data card: atlas pose and histology registration

## Scope

This card describes data roles for the frozen baseline and proposed unified-model study. Counts and hashes for a new benchmark remain pending until collected and frozen; no uncollected cohort is represented as available.

## Data inventory

### Allen CCFv3

- **Role:** atlas template, foreground, ontology labels, exact synthetic pose/flow supervision.
- **Resolution used:** 25 µm.
- **Files:** `average_template_25.nrrd`, `annotation_25.nrrd`, structure query/lookup and bregma conversion metadata.
- **Frozen hashes:** template `e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b`; annotation `c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42`; labels `9f8bc9f952a251a8f07be11c9536ff0fbf28f802892cfc39ee45e66c99a8fe3a`; query `5347daf90e02ac1d1cfcbf9c8af86ff23a2fb32cd7e7a2ba2881951931286dbd`.
- **Risks:** one population reference cannot encode all individual variation; label granularity and boundary uncertainty affect region metrics.
- **Provenance:** Allen Institute terms/citation policy; runtime hashes are maintained by `setup_runtime.py` and release metadata.

### Allen Connectivity Product 5 registered sections

- **Role at baseline:** trusted real-image pose supervision, validation and locked test for AtlasPose.
- **Unit:** experiment/animal; serial sections stay together.
- **Strength:** block-face registration gives substantially stronger plane labels than ordinary slide-mounted affine metadata.
- **Limitation:** registered pose does not imply exact dense individual-to-template correspondence.
- **Frozen inventory:** 132,442 training sections from 2,441 specimens; 6,513 validation sections from 122 specimens; 7,214 test sections from 133 specimens; 1,400 previously consumed DeepSlice-comparison sections from 10 specimens. Splits are by specimen.
- **Development status:** the warm-start AtlasPose checkpoint has already learned from the training specimens, and all existing validation/test/comparison results have already been inspected. Those sets are historical and regression evidence for the joint model, not a new publication-grade hidden test.

### Allen Product 8 slide-mounted sections

- **Role at baseline:** diagnostic only.
- **Reason:** specimen-wide pose offsets produced failed subgroup diagnostics.
- **Prohibition:** cannot affect training, model selection, calibration or release without a new, documented curation protocol.

### DeepSlice published data

- **Role:** public comparator reproduction and human-consensus reference.
- **Composition:** seven unseen slide-mounted experiments aligned by seven operators in the publication, plus published predictions; additional held-out S2P comparisons are used only under their documented availability.
- **Leakage status:** previously inspected in this project; not a pristine hidden final set.
- **Source:** Figshare DOI `10.25949/22802411` and publication DOI `10.1038/s41467-023-41645-4`.

### Local session 722

- **Role:** workflow and real-case diagnostic QA.
- **Leakage status:** repeatedly inspected and used for development.
- **Prohibition:** not an independent validation cohort and not evidence of general performance.

### Planned new real benchmarks

- **Pose target:** 30--50 animals, at least three laboratories, multiple acquisition/staining conditions and deliberately sampled coronal, sagittal, horizontal and extreme-oblique cuts, approximately 500--1,000 sections; final N by pilot power calculation.
- **Dense target:** 150--250 sections from at least 20 animals with blinded landmark and boundary annotation.
- **Probe target:** exact digital phantoms and an independent real Neuropixels cohort.
- **Status:** not yet collected/frozen.

## Synthetic data

The frozen v2 generator covers only an oblique-coronal AP/tilt chart and is
retained for historical diagnostics. A side-by-side v3 path generates every
brain-intersecting plane on demand from immutable manifests. Its reference
measure uses equal-area antipodal plane normals, uniform in-plane roll and
uniform valid offset over projected annotation support. Each sample records a
structured 3-D frame and exact QuickNII O/U/V, source-asset hashes, RNG stream,
rejection attempts, support measures and synthetic-realization identity;
animal/specimen IDs are nullable for atlas data and exact for real data.
Samples also include forward/inverse deformation, label maps, visible/damaged
masks and appearance metadata. Appearance is grayscale and includes clean
through severe tone, illumination, tiling, vignette, blur, noise, speck,
blowout, bubble, tear, missing-tissue and occlusion conditions.

The exact synthetic test uses independent transformation/artifact implementations. Different random seeds alone are insufficient independence.

The existing 8,192-case AtlasPose synthetic test has already been inspected and is development/regression evidence. The legacy dense-registration v2 sealed generator remains unconsumed and may be run once as a warp-only benchmark after candidate freezing, provided its generator and evaluator remain byte-identical. A new end-to-end joint locked test receives a separately versioned generator/evaluator contract and a hidden seed generated only after model freezing.

## Splitting and leakage control

- Split real data by animal/experiment, never by section; synthetic atlas
  realizations retain explicit source identity and disjoint hash-bound manifests.
- Serial neighbours, channels, crops and augmented descendants inherit the parent split.
- Hash exact files and audit perceptual near-duplicates.
- Validation guides selection; hidden test is consumed once after freezing.
- Public and previously viewed cohorts are labelled development/diagnostic.
- Record every exclusion and preserve pre-exclusion counts.

## Annotation quality

Real pose consensus uses multiple blinded experts and reports leave-one-rater-out variability. Dense points may be marked unidentifiable; missing tissue is invalid correspondence. Consensus procedure, ontology depth and boundary policy are frozen before evaluation. Surgical plans are priors, not ground-truth trajectories.

## Bias and limitations

Likely biases include C57BL/6-derived reference anatomy, adult-brain emphasis,
over-representation of Allen acquisition pipelines, whole-section input, clean
block-face geometry and user-selected visible masks. Performance must be
stratified by laboratory, appearance, equal-area orientation cell, physical
offset, visible support and damage. Pathological or genuinely unusual anatomy
must not be judged solely by how completely the model can deform it into a
normal atlas.

## Access and licensing

Raw Allen and third-party images remain governed by their upstream terms. This repository should distribute manifests, derived metadata and model artifacts only when permitted. Every released dataset/model version includes source URLs, access dates, licenses/terms and cryptographic hashes.
