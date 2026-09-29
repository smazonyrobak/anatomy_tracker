# TRAIN exposure census and fixed1024-section expansion

CPU-only completed-lineage census: imagekey4000→outline-A6000→bridge-B8000→
rehearsal-C10000. Discarded sibling arms and independent native controls are not
counted as training exposure of this model. No model weights, image pixels,
development evaluation outputs or benchmark data were used. The shared frozen
metadata manifest was enumerated, with expansion selection restricted to TRAIN.

## Actual exposure, not optimizer-step count

| Cumulative endpoint | Generated presentations / distinct observations | Distinct eligible positive catalogue cells | Normal-offset planes / normals covered |
|---|---:|---:|---:|
| Imagekey4000 | 32,000 / 32,000 | 35,000 / 98,304 | 6141 / 384 |
| A6000 | 48,000 / 48,000 | 50,125 / 98,304 | 6142 / 384 |
| B8000 | 56,000 / 48,000 | 50,125 / 98,304 | 6142 / 384 |
| C10000 | 64,000 / 48,000 | 50,125 / 98,304 | 6142 / 384 |

The catalogue has6144 normal-offset planes×16 rolls=98,304 full-frame cells and
384 normals. Eligible-cell counts include generated and frozen training targets.
All5120 frozen rows/fullframes were already presented by step4000; they received
64,000 cumulative presentations by C10000. Total synthetic presentations are
128,000, but only48,000 distinct generated observations plus5120 fixed rows.

The archived drivers/traces show imagekey uses generated indices0:32000,
outline-A32000:48000, bridge-B32000:40000 and rehearsal-C40000:48000. Thus B/C
added **zero new generated cells, appearance draws or PSF draws**. At C,49.01%
of catalogue cells have never been eligible positive targets. Nevertheless all
384 normals and6142/6144 normal-offset planes have positives: claiming that most
plane directions are unseen would be false. Missing roll/frame coverage and
sparse appearance repetitions remain genuine limitations.

By C,97,602 cells appeared somewhere in a sampled candidate bank, but each query
was optimized against only≈64 cells before B and≈48 thereafter, versus98,304 at
evaluation. Local negatives were chosen geometrically, not by learned scoring:
median normal difference≈7.8°, median offset≈601µm. Across stages37–38% of
synthetic queries had a candidate treated as negative despite meeting the
evaluated10°/500µm plane tolerance, because its finite-frame RMS exceeded1mm.
That is valid for a full-frame discrimination task, but differs from coarse
plane-capture ranking. These counts establish exposure/objective differences,
not that either one caused the≈38° synthetic MAP error or that model capacity is
sufficient. More data alone is not demonstrated to fix it.

A substantive next learning control would finish the remaining50,304 generated
catalogue cells, then use fresh whole-cycle appearances/PSFs/brush modes, while
testing current-model TRAIN-query high-scoring false candidates against the same
uniform-negative schedule. Preserve valid near-positive treatment, uniform
coverage and full-frame conditional information; never mine development queries.
This targets exposure/ranking without a result-driven LR sweep. It is a proposal,
not a launched experiment or justification to relax existing gates.

## Existing real data and bounded expansion

B/C each presented16,000 real queries, always the same256 images/58 donors;
244 images had common matched/canonical anchor eligibility. Each stage used
8000 matched and8000 canonical chart choices before eligibility weights.
The selected TRAIN normals are0.487–11.628° from the AP axis, with maximum
pairwise antipodal span13.2404°: more sections from these experiments do not
create arbitrary-plane real coverage or additional independent donors.

Frozen metadata:
`I:/AnatomyTracker/data/allen_real_development_20260928/metadata/sections.jsonl`
contains8952 sections, of which8113 are TRAIN from58 donors (136–140 per donor).
The current256 cover3.16% of these TRAIN records;7857 additional TRAIN section
URLs are available without querying metadata again. Metadata section SHA256:
`f209370dc548e36b4a887524041b5d5c92ec6c2b91f770b5b64d8a543e4b7a9c`;
experiment metadata SHA256:
`67b19f91610169371e4a824b7e1095e3ab2e1b9a5720a7c04a0cfb99afbf401b`.

Root approved **1024 additional sections**, not all7857. Existing deterministic
donor-round-robin selection of1280 TRAIN records, minus the original256 exact
section IDs, yields38 donors×18 additions and20 donors×17. Ordered additional
IDs (compact JSON integer list) SHA256:
`4d1d731c9d668112bad0e140f3c5bfe5fbd5476f8c1862f2040dcd8c62e753ea`.
Donors14452,15219,15336,15439,15447,15935 remain excluded from downloads.
URLs retain official `api.brain-map.org/api/v2/image_download/<SectionImage.id>`
with the frozen equalization range and `downsample=5`. No benchmark source is used.

The original256 TRAIN JPEGs total19,922,095 bytes: proportional estimates are
79.7MB for1024 additional JPEGs and611MB for all7857, not measured future sizes.
New float32[1024,1,96,96] inputs occupy≈37.75MB. I: had1.786TB free at preparation.
This estimates storage, not the eventual downloaded sizes.

## Implementation contract and source

Root reviewed the fixed selection, existing downloader and unchanged conversion;
`training/acquire_joint_v6_allen_training_expansion.py` is enabled for a separately
committed launch. No completion or usability claim is made here. Fresh output:
`I:/AnatomyTracker/data/allen_real_training_expansion_20260929`.
It reuses `training/acquire_allen_real_histology_images.py`:
`deterministic_section_selection` and `_download_record`, preserving official
requested/returned URLs, bytes/hashes, transport metadata, JPEG decode and exact
source identities. The high-level `acquire_image_snapshot` has no append/exclude
API; the original snapshot is therefore neither mutated nor passed changed quotas.
A recorded download failure stops the run without replacing the selected section.

Existing launcher: `training/acquire_allen_real_development_20260928.py`;
metadata implementation: `training/allen_real_histology_metadata.py`;
conversion reference: `training/prepare_joint_v6_allen_training_inputs.py`, SHA256
`f17022e971c9054b80e4bd1d61b2994495e425948b8003147d378ad86070db33`.
Downloader SHA256:
`1a3d935c3e829391f4e6d7b6f56e0d099b0065a8769fd98fb3e9bc9201e139e4`.

Conversion remains red/255, acquisition-centred12mm96², identical Gaussian
antialias/bilinear resampling, acquired-border median padding, no segmentation,
outline or photometric tuning. It retains model→full-pixel→downloaded-pixel
transforms, Allen2D/3D affine, thickness provenance and+12.5µm AP/DV/ML model
coordinate shift. Geometry/IDs do not become encoder inputs. These are weak
upstream registration references, not dense deformation or calibrated labels.

The union1280 index links original256 and new1024 arrays by absolute file path
and row index, and links original JPEG locations without copying/rewriting them.
Completion stores planned IDs, manifests/failures, raw byte hashes, per-row
affines/IDs, array hashes and terms. The six development donors are not used to
choose sections or preprocessing. Historical benchmark-exclusion uncertainty
remains inherited: this is internal training/development, not untouched final
validation. No train/validation animal split is changed.

## Reproducible census

`I:/AnatomyTracker/runs/joint_v6_lineage_training_exposure_001/analyze_exposure.py`
SHA256 `7157310ea411de3eb4b62afab34e7b36f66dcf6c0def754aed6e99eaa4fc602d`;
`exposure.json` SHA256
`a89718feae75db03c8e874fcbe5ad80ebf8257456741d6053f103294844eaf5d`.
The flat CPU analysis preserves source/schedule/trace/metadata hashes. It reads
only label, eligibility and pose tensors from the frozen TRAIN tensor pack (no
image tensor use), plus atlas catalogue geometry, to count target exposure and
geometric candidate relations. Native signed-pose run70497 remains untouched.
