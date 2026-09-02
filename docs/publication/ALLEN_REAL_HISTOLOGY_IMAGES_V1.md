# Allen Product-5 bounded image acquisition v1

## Scope and data role

This stage turns a verified donor-bound Allen metadata snapshot into a small,
immutable real-image development source. It is limited to Allen Mouse Brain
Connectivity Atlas Product 5 and inherits each section's exact `Donor.id`
animal key and development split. It does not accept public-comparison,
external-laboratory or final-test rows.

Product-5 sections are used for real acquisition/appearance information only.
Their predominantly canonical coronal geometry cannot provide arbitrary-plane
coverage; full plane normal, offset, roll, finite thickness and deformation
remain supervised by the CCF and provenance-bound synthetic generator. No old
model weight, feature, embedding, prediction or pseudolabel is an input.

## Acquisition and verification contract

`training/acquire_allen_real_histology_images.py`:

- accepts only a verified metadata snapshot on I:;
- requires image output to resolve to I: and remain outside every Git
  worktree;
- selects only metadata-eligible `development_train` or
  `development_validation` rows;
- deterministically ranks donors by SHA-256 and samples round-robin across
  donors before taking a second section from any donor;
- uses the exact range/downsample URL frozen in the source section record;
- preserves the response URL, HTTP metadata, exact unmodified JPEG bytes,
  SHA-256, byte count and decoded dimensions;
- binds each image to the canonical source section record, experiment record,
  and raw per-experiment API-response hash; and
- records source, eligible, metadata-excluded, selected, not-selected,
  downloaded and transport/decode exclusion counts before issuing an immutable
  whole-tree receipt.

`training/verify_allen_real_histology_images.py` is an independent verifier. It
does not import selection or hashing logic from the acquisition module. It
recomputes the deterministic section selection, source lineage, all metadata
and image hashes, the complete metadata and image artifact sets, pre/post
counts, official HTTPS API hosts, decoded JPEG properties, development-only
splits, and data-role/terms constraints.

## Official source and terms

The exact image service is documented by the
[Allen image-download documentation](https://brain-map.org/support/tutorials/downloading-an-image).
Source experiment/section semantics and registration metadata are documented
by the
[Allen Brain Connectivity API documentation](https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas).
The bytes remain subject to the
[Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-of-use)
and [Citation Policy](https://alleninstitute.org/legal/citation-policy).

No SPDX dataset license is asserted. The raw JPEGs are local development data,
remain outside Git, and are not approved for redistribution by this manifest.

## Three-image official-API dry run

The bounded dry run is stored at
`I:\AnatomyTracker\data\allen_real_histology_images_v1_sample_20260902`.
It independently verifies against metadata manifest
`4f0a71416705494ee93f01171c11ddce0e51fa41f2a0008dc1d081c1ff6917df`.

- Source population: 420 metadata-eligible sections from 3 donors; 0
  metadata exclusions.
- Selection: 3 sections, one from each donor; 417 eligible sections not
  selected.
- Download result: 3 valid JPEGs, 265,847 raw image bytes; 0 transport/decode
  exclusions.
- Development splits: train only, because all three animals in the bounded
  metadata sample hash to development train. This is not a validation cohort.
- Image snapshot: 6 files and 275,415 total bytes including manifests.
- Manifest SHA-256:
  `62ff33e16745a7101dc4f00da77afa86d74e4464f0f811611894b3dcbdc26c4d`
- Receipt SHA-256:
  `4cd43daa7a21eae0e5e4bcbc30020588494561e77f80f58c1e1d24237e88a22d`
- Image-manifest SHA-256:
  `3d3ce0ee5eb1ccaa02f942fd8d295d9785dcf432b5c927d41ac53b64048d4fa2`

Selected exact identities and image hashes:

| Donor | Specimen | Experiment | SectionImage | JPEG SHA-256 |
|---:|---:|---:|---:|---|
| 14451 | 711104 | 100140949 | 102125385 | `598138413dbcc2aa2e51074cf5c7ec6955caa9c2848684416c7039fc70beb367` |
| 14383 | 710918 | 100140756 | 102119369 | `0b0fa469c4909b67c861b71321152afb5b9b4957e54573f97c60e602517d1ed2` |
| 14453 | 711106 | 100141214 | 102127264 | `cf0fcfaa34b27dc84f1d2888dad995411669139a1b5341b9e325706839b0e694` |

The images are downsample-5, equalization-windowed RGB JPEG responses from the
official image service (1249 x 937 pixels). They are useful as a transport,
lineage and appearance-path proof, not evidence of model accuracy or domain
coverage. Product-5 fluorescence, viral signal, mouse-line distribution and
Allen acquisition conditions remain materially different from routine lab
histology, so later real-lab external validation is still required.
