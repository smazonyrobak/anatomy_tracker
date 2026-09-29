# Coherent acquisition views 001 — frozen CPU audit

**Integrity passed; small TRAIN-only acquisition-frame supplement, not a model
qualification.** All 128 physical sections and 384 paired observations were
retained. The eight accepted synthetic TRAIN anatomies are reused, not eight new
biological animals. No development generation, GPU inference, atlas rerender,
training, or public benchmark was performed by this audit.

Frozen corpus:
`I:/AnatomyTracker/data/joint_v6_coherent_acquisition_views_001`.
Independent audit:
`I:/AnatomyTracker/runs/joint_v6_coherent_acquisition_views_001_independent_audit`.
The executable flat NumPy source is archived there as `audit_source.py`; raw
per-section and per-observation findings are `sections.jsonl` and
`observations.jsonl`. Its completed execution returned exit 0.

## Coverage and information

| Presentation | Eligible | Censored | Retained |
|---|---:|---:|---:|
| Raw | 76 | 52 | 128 |
| Exact black | 76 | 52 | 128 |
| Imperfect brush | 75 | 53 | 128 |

Eligibility remains finite support mass >=64 and visible support mass >=64;
227/384 observations qualify for geometric supervision. Raw eligible counts by
subject index 0–7 are **8, 9, 9, 9, 7, 10, 12, 12** out of 16. One brush-eroded
view of subject 5 loses eligibility. These are paired views, not 227 independent
anatomies. Each of the 58 TRAIN nuisance donors supplies two or three finite-frame
tuples; nuisance identity remains separate from synthetic subject lineage.

Of the 52 raw-censored sections, **49 have exactly zero tissue support**. Six
have every slab query outside the atlas physical extent; 43 zero-support slabs
still contain some in-extent queries. The remaining three censored sections have
positive but insufficient support, at most 14.8083 pixel-mass. Thus completely
out-of-bounds sampling explains only part of censoring; in-bounds empty atlas
locations are not evidence of visible brain.

Censored versus eligible sections have mean out-of-atlas slab-query fractions
**91.88% versus 53.28%**, and mean absolute sampled box-offset/radius
**0.819 versus 0.374**. This associates censoring with peripheral draws. Median
new/full-box chart area is 0.591 versus 0.631, while median tangent shifts are
1.904 versus 1.921 mm: the stored evidence does not support blaming a larger
translation alone. Across all sections, mean tangent shift is 1.927 mm and mean
spans are 12.948 and 14.097 mm. Without extra rendering this audit cannot separate
a truly empty plane from a finite field of view missing tissue; it does not
discard or resample either. Entire-slab outside counts use physical atlas extent,
not the slightly smaller voxel-centre interpolation domain.

## Independent provenance and geometry checks

The audit authenticates the parent plan records and their artifacts, frozen
source archives, catalogue/support origin, GPU mapping-equivalence receipt,
TRAIN-only nuisance geometry, and all section metadata/array hashes. It
reconstructs the full donor/row schedule and each plane/thickness/reflection seed
branch. The full-RP2 normal, roll and conservative box/slab offset are unchanged
by chart replacement; borrowed joint centre/span/shear tuples use physical
subject micrometres without an extra scale or half-voxel shift.

Independent x/96–y/96 frame reconstruction agrees with saved subject queries
within **7.28e-12 um**; plane-equation error is **3.79e-12 um**. Spatial-only
reflection restores canonical coordinates exactly. Independent full-canvas
least-squares fitted coordinates differ by at most **2.75e-10 um**. Full-frame
centre and actual pixel centroid are checked separately. Paired image pixels,
boundary/availability channels, exact-black and imperfect masks, visible support,
PSF labels/occupancy, identities, and censoring agree. Float32 support-reduction
order differs by at most 0.0004883 pixel-mass; the arrays themselves agree exactly.

The accepted 3D flow and saved forward-mapped support origins were authenticated,
not numerically replayed. Atlas interpolation and label lookup were not rerun.

## Exact-target representability screen

An independent full-canvas affine-free centre residual and centre-anchored,
PSF-weighted least-squares director were fitted to the **actual saved 3D slab
coordinates**, not to an image warp. Every section, including censored sections,
fits the current component caps and conservative bilinear derivative bound:

| Quantity | Maximum over 128 sections | Current limit |
|---|---:|---:|
| Absolute local centre-residual component | 141.3991 um | 200 um |
| Absolute local director-delta component | 0.131603 | 0.2 |
| Derivative Frobenius bound | 0.168705 | 0.35 |

Residual affine moments are <=3.35e-11 um. The centre-plus-linear-director slab
approximation has maximum point error **0.24811 um**, maximum full-canvas
PSF-weighted RMS **0.03417 um**, and maximum eligible visible-support/PSF RMS
**0.04637 um**. Exact slab targets remain the supervision; no approximation
replaces them. Curvature is real: eligible raw visible normal residual RMS
averages **32.96 um**, reaching **52.93 um** relative to the fitted plane.

These bounds establish a suitable full-resolution representative, not exact
24x24 decoder expressivity, finite tanh-logit behavior, learnability, or model
accuracy. The corpus supplies only 75–76 informative physical views across eight
already-used anatomies; keep full censored coverage and the unchanged held-subject
checks, and do not infer generalization to new finite-FOV acquisitions from this
TRAIN-only audit.

## Exact receipts

- Corpus completion SHA256:
  `c0fb0763bb520adf3f372385ad2b1c5c52df6887db855de8a7f3ed99123bf5a5`.
- Independent `audit.json` SHA256:
  `701b41470fd9b912f9af0fb12a9665f636df3ea3fdabfbbd4161cad6a0cf0426`.
- Archived audit source SHA256:
  `f1affa87d5648525902861288f5fc364dc512e92b56759a747c82b890ac9ffe8`.
- Per-section results SHA256:
  `1c3215fb72aecff52322cd670ac050bd0cbdc4c29ca4c9ba6462c9419b6b0096`.
- Per-observation results SHA256:
  `fe1b5654b644805dc5408fe14ac2b19c90cbf6be3b1b1714bd35373077bb2aee`.

The audit JSON retains the exact source/parent/nuisance/section file hashes.
No claim of calibrated uncertainty, deployment readiness, or benchmark superiority
follows from this data-integrity and geometric-capacity result.
