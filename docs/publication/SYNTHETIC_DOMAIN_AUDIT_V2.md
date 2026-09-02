# Arbitrary-plane synthetic acquisition-domain audit v2

Date: 2026-09-02  
Audited base commit: `bbb2b45c272188bfe8583ae34f7cafcd6daaced3`

## Scope and boundary

This is a source- and contract-level audit of the active finite arbitrary-plane
synthetic path. It asks whether the generator exposes a randomly initialized
model to the main geometric, acquisition and preparation variation expected in
real histology. It is not a benchmark, model result or claim that the chosen
augmentation frequencies match a target laboratory population.

The audit and its tests ran CPU-only in a separate `I:`-resident worktree. No
running-pilot output, temporary artifact, checkpoint, receipt or qualification
result was opened, hashed, loaded or modified. The additions below are a
candidate **next generator iteration**. They do not amend or reinterpret the
frozen pilot.

## Audit result

| Requirement | Base state at `bbb2b45` | Audit decision |
|---|---|---|
| Any brain-intersecting plane | Implemented: surface-area-uniform normal on RP2, full roll and offsets over brain support; marginal raster support is retained rather than redrawn | Keep |
| Finite physical thickness | Implemented: authenticated finite-PSF slab observation, with centre-plane targets kept separate from slab-only evidence | Keep |
| Nonrigid processing deformation | Implemented: one positive-Jacobian map with physical loss and occlusion separated from correspondence | Keep; do not let appearance artifacts become deformation targets |
| Contrast/stain variability | Partly implemented: label-conditioned grayscale contrast, polarity, gamma, gain, offset and bias fields | Keep for the grayscale model; defer RGB/channel- and modality-physical rendering to a measured ablation |
| Blur, resolution and noise | Implemented: blur, anisotropic resolution and additive noise | Keep |
| Raw/acquired backgrounds | Gap: finite-slab raw views inherited only a dark `0--0.25` background prior | Add named dark-field-like, brightfield-glass-like and neutral scanner-stress families |
| Unequal illumination | Local bias existed, but no explicit whole-frame gradient/vignette shared by tissue and glass | Add a named global gradient/vignette stream |
| Compression | Gap | Add deterministic grayscale block-DCT coefficient quantization after damage and before optional brush masking |
| Tears, holes, missing cortex and occlusion | Implemented with invalid-correspondence masks | Keep |
| Thin scratches/streaks and mounting bubbles | Gap | Add explicit appearance-invalid line and annular families |
| Tissue folds | Partial: a bright/doubled-looking strip exists, but it is not a physical overlap with two possible atlas correspondences | Keep it labelled as an appearance-invalid stressor; do not claim true fold simulation |
| Accurate smart brush | Implemented with an exact-zero exterior | Keep |
| Imperfect smart brush | Implemented with deterministic outline perturbation and an exact-zero exterior | Keep |
| No brush/mask | Implemented: raw acquired background is retained and no automatic segmentation is required | Keep as a mandatory training track |
| Animal/specimen/experiment/section provenance | Exact IDs existed in finite curriculum rows, but a low-level synthetic artifact bound only parent provenance | Bind the six-field lineage tuple into each synthetic artifact and its final identity |
| Prior-model independence | No checkpoints, previous models, pretrained features or pseudolabels are dependencies | Keep |

## Literature rationale

- DeepSlice trained with multiple real stain/acquisition families and synthetic
  noise, pixel dropout and warping, while converting inputs to grayscale. This
  supports broad grayscale appearance randomization for the current input
  contract, but it does not justify pretending that one fixed background covers
  brightfield and fluorescence. DeepSlice also reports deformation as a source
  of failure. [Carey et al., 2023](https://doi.org/10.1038/s41467-023-41645-4)
- SynthMorph shows that diverse label-conditioned contrasts and synthetic
  deformation can force contrast-invariant registration features without
  acquired training images. Its MRI evidence motivates the mechanism, not
  histology-specific parameter frequencies. [Hoffmann et al., 2022](https://doi.org/10.1109/TMI.2021.3116879)
- Scanner, staining, tissue-processing and acquisition differences create
  cross-centre histopathology shift, and domain-specific augmentation improves
  generalization. [Faryna et al., 2024](https://doi.org/10.1016/j.compbiomed.2024.108018)
- Focus degradation and JPEG compression are realistic digital-pathology
  stressors and can change model performance. The implementation uses a
  deterministic grayscale transform-codec approximation rather than claiming
  byte-level emulation of a particular scanner. [Schömig-Markiefka et al., 2021](https://doi.org/10.1038/s41379-021-00859-x)
- Histology preparation commonly creates tears, folds, wrinkles, scoring/knife
  lines and air bubbles. [Taqi et al., 2018](https://doi.org/10.4103/jomfp.JOMFP_125_15)
- Tears, holes and tissue loss specifically complicate multimodal histology
  registration, supporting their treatment as missing/invalid evidence rather
  than ordinary elastic displacement. [Feenstra et al., 2024](https://doi.org/10.1117/1.JBO.29.6.066007)

These sources support which variation families must be represented. They do
not establish universal probabilities or severity bounds for mouse-brain
histology, so the current mixture remains an explicit engineering prior.

## Implemented next-iteration changes

`training/arbitrary_plane_synthetic_generator.py` now uses schema/algorithm v2
and adds:

- named finite-background families with separate base-intensity ranges;
- optional global illumination gradient and vignette applied consistently to
  tissue and acquired background;
- optional deterministic block-DCT transform compression;
- thin scratch/streak and mounting-bubble-ring appearance-invalid events;
- exact `animal_id`, `specimen_id`, `experiment_id`, `synthetic_animal_id`,
  `section_id`, `split` lineage in the resolved config, artifact receipt and
  final synthetic identity.

The finite pose and finite joint curricula pass their already required lineage
into every paired synthetic descendant and reject disagreement. Low-level
legacy calls cannot invent missing grouping: they explicitly record a
`partial-parent-fallback` with null synthetic-animal and section IDs. The v1 RNG
seed domain is deliberately retained, and all additions use new named streams,
so legacy non-slab arrays remain byte-exact while v2 artifacts have new
identities.

Focused tests force each new background family, force compression on/off,
verify exact replay, preserve exact-black exteriors for accurate and imperfect
brush modes, retain raw backgrounds when the brush is absent, exercise all
seven damage families, reject lineage conflicts/tampering, and check both
downstream finite curricula and row caches.

## Deliberately deferred gaps

1. Do not add a learned stain-transfer or style network: it would add an
   unnecessary learned dependency and conflict with the standalone cold-start
   requirement. RGB or multichannel modality rendering should be tested only
   if eligible real development data show a grayscale performance gap.
2. Do not represent a true tissue fold as a single-valued deformation. A fold
   can contain overlapping layers and ambiguous correspondences. Until a
   separately verified multi-layer model exists, fold-like pixels remain
   appearance-invalid and excluded from dense supervision.
3. Do not require automatic foreground segmentation. Accurate, imperfect and
   absent-outline descendants remain paired views of the same latent section.
4. Do not claim that current probabilities are population estimates. Measure
   stain/modality, background, codec, blur, tear, bubble and brush-error
   distributions on eligible training/development acquisitions, then freeze a
   revised mixture before large training.

Before promoting v2, perform a blinded stratified montage review and a small
real-development stress test by acquisition family. Preserve exact animal
grouping and do not access untouched final-test animals. Full DeepSlice,
expert-assisted and external-lab benchmarking remains deferred until the model
and generator stabilize.
