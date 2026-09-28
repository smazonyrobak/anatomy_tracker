# Allen real-image development cohort — 2026-09-28

Acquisition completed at `2026-09-28T13:57:28.573608+00:00` under
`I:\AnatomyTracker\data\allen_real_development_20260928`. The existing independent
image verifier completed successfully before the acquisition process exited zero.

| Partition | Donors/specimens/experiments | Downloaded sections | Sections per donor |
| --- | ---: | ---: | --- |
| Development train | 58 | 256 | 24 donors × 5; 34 donors × 4 |
| Development validation | 6 | 64 | 4 donors × 11; 2 donors × 10 |

The metadata population contains 64 donors, 64 specimens, 64 experiments and
8,952 eligible sections, with zero metadata exclusions. Selection downloaded
320 JPEGs with zero transport/decode exclusions: **25,200,426 image bytes**.
The complete cohort, including metadata, raw API/documentation responses and
receipts, contains 398 files and **43,526,562 bytes**.

The inherited hash split uses exact Allen `Donor.id`; image selection is the
existing deterministic donor round-robin. Independent intersections of selected
donor, specimen, experiment and section IDs were all empty. Validation donor IDs
are `14452`, `15219`, `15336`, `15439`, `15447`, and `15935`; their section counts
are 11, 10, 11, 11, 11, and 10 respectively. Exact identities and image hashes are
in `images/images.jsonl`; raw API responses and alignment metadata are in
`metadata/`. No earlier model weights, embeddings or predicted labels were used.

Exact receipts:

- Metadata receipt file SHA-256:
  `34d5fc1e8b75398b0d8393059327402dbe1a02d3e15235293c3c5054a49244f7`
- Image receipt file SHA-256:
  `653be9c2357c1461a67e2d997ebd210c622931aceefa3dcd257d44146d4171fa`
- Cohort summary file SHA-256:
  `94ae9603dfed354ecb1c4aa82f57ae79f81657448542878b50b4941f4c87302f`
- Acquisition script SHA-256:
  `3f3a8130bb3302416ad38621278a23be2b2ed1ab0cd7a92fbe1c6b2bbca08fd8`

Images are official Allen Product-5, downsample-5, equalization-windowed RGB JPEGs.
Raw URLs, returned URLs, byte hashes and the applicable Allen Terms of Use and
Citation Policy snapshots are preserved by the existing acquisition functions.
Image bytes remain outside Git; this acquisition does not approve redistribution.

This cohort is for appearance training and internal diagnostics only. Repository
records disclose earlier use of the public DeepSlice brains and a local Allen
comparison cohort, but no joinable historical Allen-donor exclusion list was
available in the inspected I: data/manifests. Consequently these images cannot
support an untouched final-generalization claim. No benchmark images or expert
benchmark labels were opened during acquisition. Product-5 coronal fluorescence
also does not establish arbitrary-plane or ordinary laboratory-histology accuracy.

## Small next experiment

The historical `registered_section_dataset.py` automatically masks, reorients and
canonicalizes images through the legacy AtlasPose preprocessing path. Reusing that
path would impose segmentation and orientation assumptions absent from the new
model's input contract. The current synthetic appearance generator instead uses
explicit fluorescence/brightfield response, illumination, noise and independently
acquired background, but its parameter ranges are engineering priors.

First run a raw-image proposal diagnostic on the six development donors after the
current experiment finishes. Use the recorded background-fluorescence red channel,
fixed intensity scaling and a recorded physical crop/pad/resample into the model's
12-mm field; preserve the raw exterior and supply zero outline/availability. Any
physical resampling must retain its exact source-pixel transform. Compare predicted
planes with the retained upstream Allen registration metadata only as a coarse
diagnostic; those affine registrations are not expert dense-warp truth.

Then compare two short continuations from the same standalone checkpoint and
identical synthetic row schedule: unchanged appearance versus zero-preserving
gain/gamma augmentation informed only by the 58 training donors. Retain arbitrary
plane and smart-brush training mixtures. Measure paired change in synthetic
development NLL/top-K/plane errors and donor-macro real diagnostic errors. Do not
fit appearance parameters on the six development donors, treat image count as
animal count, or infer deformation correctness from agreement with Allen affine
metadata. This is a small domain experiment, not a new model or full benchmark.
