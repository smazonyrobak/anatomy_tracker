# Allen real-histology metadata source v1

## Decision

The standalone arbitrary-plane model may use Allen Mouse Brain Connectivity
Atlas Product 5 sections as a real acquisition/appearance source, but not as
the source of arbitrary-plane geometric coverage. Product 5 is predominantly
registered coronal serial two-photon data. Arbitrary plane normals, offsets,
roll, finite thickness and deformation therefore remain supervised by the
provenance-bound CCF/synthetic generator.

This path has no dependency on an earlier model, feature extractor, embedding,
prediction or pseudolabel. Allen registration metadata is retained as upstream
metadata and for bounded diagnostics; it is not evidence that the real source
spans arbitrary cutting planes.

## Official sources

- Allen Product 5 metadata: `https://api.brain-map.org/api/v2/data/query.json`
  and per-experiment `SectionDataSet` API responses. The frozen query requires
  `failed=false`, `reference_space_id=9`, `plane_of_section_id=1`, and Product
  ID 5.
- [Allen Brain Connectivity API documentation](https://brain-map.org/support/documentation/api-allen-brain-connectivity-atlas):
  Product 5 uses serial two-photon tomography; the red background-fluorescence
  channel supports registration; each `SectionImage` has an `Alignment2d` and
  each `SectionDataSet` has an `Alignment3d` into ReferenceSpace 9 in PIR
  coordinates.
- [Allen image-download documentation](https://brain-map.org/support/tutorials/downloading-an-image):
  `SectionImage.id` identifies image downloads, downsample is a power-of-two
  level, and the six-value range is derived from the experiment Equalization
  record.
- [Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-of-use)
  and [Citation Policy](https://alleninstitute.org/legal/citation-policy).
  The acquisition records these as governing documents; it does not assert an
  SPDX dataset license or commercial-use permission. Raw-image redistribution
  requires a separate terms review.

The API does not expose a Product-5 release-version field. Each acquisition is
therefore versioned by schema, UTC access time, exact request/response URL, and
SHA-256 of every saved response body. The legal and documentation pages are
also snapshotted and hashed with the metadata.

## Existing-code and I:-data audit

The repository already contains a historical Product-5/8 acquisition path in
`training/acquire_allen_s2p.py`, and `training/registered_section_dataset.py`
can consume its experiment/section manifests. Those records preserve
experiment, specimen and section IDs, but the historical acquisition assigns
splits by `Specimen.id`. The official API exposes `Donor.id` above the
specimen, and one donor can be associated with more than one specimen or
experiment. The historical path also includes previously consumed comparison
identities. It is therefore not the source contract for this standalone model.

The supported model-development data roots audited on I: contained the Allen
CCF volumes and synthetic arbitrary-plane rows, but no Product-5 image-byte
cohort or current donor-keyed real-section manifest. Tracked historical model
reports describe prior real-section use, but those learned artifacts and old
splits are excluded from this model.

## New metadata-only contract

`training/allen_real_histology_metadata.py` writes:

- raw official API responses and official documentation/legal pages;
- `experiments.jsonl` with exact Allen `Donor.id` (`animal_id`),
  `Specimen.id`, `SectionDataSet.id`, product/reference/plane IDs, channel and
  equalization metadata, alignment metadata, source URLs and response hashes;
- `sections.jsonl` with inherited animal/specimen/experiment identity,
  `SectionImage.id`, section number, resolution, alignment, exact future image
  URL, eligibility/exclusion reason, and explicit `image_status=not_downloaded`;
- a manifest fixing data roles, development-only donor split policy, source
  semantics and counts; and
- a receipt binding every file by path, byte count and SHA-256.

The split key is namespaced `allen-donor:{Donor.id}` and is assigned only from
`Donor.id`. Every specimen, experiment, serial neighbour and future image
descendant inherits that donor split. Records without a donor ID are
ineligible. No final-test animals are defined or accessed by this development
source.

## Bounded live dry-run

A metadata-only official-API dry-run was written to
`I:\AnatomyTracker\data\allen_real_histology_metadata_v1_sample_20260902`.
It contains 3 donors, 3 specimens, 3 experiments and 420 eligible section
records, totalling 1,699,161 bytes across 12 metadata/source files. It contains
zero image files and downloaded zero image bytes.

- Experiments: `100140756`, `100140949`, `100141214`
- Donors/animals: `14383`, `14451`, `14453`
- Specimens: `710918`, `711104`, `711106`
- Manifest SHA-256:
  `4f0a71416705494ee93f01171c11ddce0e51fa41f2a0008dc1d081c1ff6917df`
- Receipt SHA-256:
  `f5152f4aff7edab78a691e845fbf019052adb8126e472590c7cfc57609053d0c`
- Experiment-manifest SHA-256:
  `a2b711ad6d8e313314fd7826312e67f1402d70d62429fcaf67f81bf49396b55f`
- Section-manifest SHA-256:
  `d87e0e8f88dee3d5cffa26480045f8d5cf1cdf57a0473db4e7794448c4223f76`

All three dry-run donors happen to hash into the development-train split; this
small sample is a contract check, not a training cohort or validation panel.

## Training role and limitations

Real Product-5 sections can teach actual section support, illumination,
texture, acquisition noise and tissue-loss/background patterns. Exact-black
and imperfect smart-brush modes remain synthetic/augmentation variants, so
real input never requires automatic segmentation. Product-5 fluorescence,
viral projection signal, mouse lines and Allen acquisition conditions do not
span routine laboratory histology; real lab histology remains necessary for
later external validation.

The current metadata deliberately downloads no images. Before real-image
training, a separately frozen step must download only selected donor-split
development images to I:, hash exact bytes, bind them to these source-response
hashes, preserve pre-exclusion counts, and keep raw Allen bytes out of Git.
The bounded dry-run did not access any public comparison benchmark, external
laboratory cohort, or final-test data.
