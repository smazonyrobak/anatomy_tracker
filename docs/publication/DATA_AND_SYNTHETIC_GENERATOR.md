# Data provenance and synthetic generator

## Source data

### Allen CCFv3 25 µm

`average_template_25.nrrd` supplies the anatomical intensity volume and `annotation_25.nrrd` supplies brain support and Allen structure identifiers. Atlas files, ontology/query table and bregma conversion constants are checksum-pinned. The CCF supports exact synthetic plane, mask and label supervision; it does not represent every individual's anatomy.

### Allen registered histology

The frozen AtlasPose baseline uses Allen Product 5 connectivity serial two-photon/block-face sections as trusted real-image pose supervision and checkpoint-selection data. Product 8 slide-mounted affine labels are diagnostic only because specimen-level offsets were observed. Product 8 cannot enter training, selection, calibration or a release gate unless a future, independently documented curation changes its role.

Product 5 contributes realistic appearance and registered plane labels. It does not automatically provide exact individual-anatomy dense deformation ground truth.

### DeepSlice public benchmark

The published human-aligned dataset and published predictions are comparator evidence only. They have already been inspected during this project's development and are not a pristine hidden test. Their original development/test brain assignment is preserved.

### Local real sessions

Session 722 and any previously opened local images are development/diagnostic cohorts. They may test workflow and reveal failure modes but cannot support an untouched generalization claim. New real benchmark animals must be independently collected and split before model development.

## Unit of splitting

Animal/experiment is the minimum split unit. Serial sections, repeated acquisition channels, thumbnails, crops and every augmented descendant inherit that unit's split. Hash-based audits reject duplicate or near-duplicate images across splits. Laboratory-held-out and acquisition-held-out subsets are separately identified.

## Unified synthetic sample contract

Each generated sample records:

- generator/version and manifest hashes;
- source atlas/data hashes;
- AP, L--R and D--V plane pose;
- in-plane rotation, scale and translation;
- fixed template, fixed labels and brain mask;
- moving grayscale section and pre-artifact image;
- exact forward/inverse maps and stationary velocity when applicable;
- visible, damaged, missing and model-input masks;
- artifact types, parameters and severity;
- positive plane and prespecified wrong-plane candidates.

Generation is on demand from immutable manifests so 500,000 training views need not become 500,000 persistent image files.

## Geometry distribution

Initial ranges follow the demonstrated AtlasPose generator unless development evidence justifies a frozen amendment:

- AP from +500 to -4500 µm with balanced 500-µm bands;
- L--R and D--V tilts from -35 to +35 degrees;
- arbitrary in-plane rotation;
- image scale from 0.5 to 1.5;
- bounded translations;
- radial, anisotropic stretch and swirl velocity components;
- local expansions/compressions, asymmetric warps and compound deformations;
- positive-Jacobian forward and inverse ground truth;
- tears, missing cortex and occlusions represented as invalid tissue rather than deformation.

Wrong-plane candidates sample very close offsets (including the 25-µm atlas step), moderate offsets and anatomically confusing distant planes. Pose ranking must remain sensitive after an optimal bounded local warp.

## Appearance distribution

All model inputs are grayscale. Include clean through severe combinations of:

- gain, offset, gamma, polarity and nonlinear tone curves;
- black through grey backgrounds and unequal hemisphere illumination;
- local over/underexposure and clipped blowouts;
- tiling seams, tile gain variation and within-tile vignette;
- noise, blur, streaks, thin scratches and bright specks;
- bubbles, tears, edge loss, polygon occlusion and edge-to-edge blackout;
- label-conditioned synthetic appearance to break dependence on CCF template intensity.

The proposed initial mixture is 10% clean, 45% mild, 35% moderate and 10% severe. Exact frequencies are frozen after a blinded visual generator audit and before large-scale training.

## Generator QA

Before training:

- inspect stratified montages with transform metadata hidden from the reviewer;
- numerically compose forward/inverse maps;
- check label resampling and validity masks;
- require positive Jacobians for synthetic ground-truth deformation;
- verify pose recovery from rendered plane anchors;
- check every corruption both alone and in realistic combinations;
- ensure the model cannot infer pose from canvas scale, padding, filenames, random seeds or artifact parameters.

## Independent synthetic test generator

The locked exact benchmark uses an independently implemented transform family, different seeds and disjoint manifests. Where training uses the repository's velocity components, testing should add independent B-spline/TPS-like or alternative diffeomorphic fields, independently generated tears/backgrounds and separate artifact textures. Results must state that all anatomy still derives from CCF and therefore cannot replace real external validation.

## Data cards and amendments

Every dataset receives source, license/terms, acquisition, animal count, section count, exclusions, intended role, split, hashes and known limitations in [`../../DATA_CARD.md`](../../DATA_CARD.md) or a linked immutable manifest. Any protocol amendment is dated, justified and marked as occurring before or after access to the affected benchmark.
