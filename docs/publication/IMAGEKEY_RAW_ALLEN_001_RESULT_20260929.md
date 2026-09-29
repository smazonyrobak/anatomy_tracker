# Image-key001 raw Allen development result

**Failed synthetic-to-raw transfer diagnostic; not a usable registration model.**
The fixed 64 acquired images from six development donors give donor-macro plane
angle **45.0506 degrees**, normal-offset error **4150.575 µm**, physical-plane
capture **0.09091 at32** and **0.21364 at128**. These compare against upstream
Allen affine metadata, not blinded anatomical landmarks or dense deformation
truth. No benchmark or biological-accuracy conclusion is warranted.

The diagnostic exited0 from commit
`c9a94c65665135bd5bc0fba4ea86138efea111df`. Frozen output:
`I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001_allen_raw`.
Its `summary.json` SHA256 is
`2394eae62ef6c64e04b41947e57ca94dd64a2c456e0bc04a58b8dc2997107b37`.
Checkpoint, full-gallery and independent-audit hashes are pinned in
[the protocol](IMAGEKEY_RAW_ALLEN_DIAGNOSTIC_PROTOCOL_20260929.md).

## Independent post-exit checks

A small flat CPU analysis authenticated77 files: saved inputs/outputs, diagnostic
source, cohort receipts, each64 source JPEGs, catalogue, whole checkpoint,
matching gallery and audit. It independently reconstructed all cell log
probabilities from saved component scores without adding any further prior;
maximum difference was `3.46474e-6`, with full posterior log-normalization error
`2.83674e-7`. Recomputing the64×128 selected component scores from normalized
query/gallery descriptors plus each representation/cell prior once agreed within
`8.16804e-6`. This is not a second model encoding or exhaustive descriptor replay.

Stable top128 indices, exact four identity fields, reference centres/normals from
the saved physical pixel affines, antipodal normal angles, sign-aligned offsets,
same-candidate captures, per-donor means and equal-donor macro means were
independently reproduced. No image exclusion or alternative-channel scoring was
introduced. Physical capture requires angle<=10degrees **and** offset<=500µm
for the same candidate; it does not certify in-plane frame or dense accuracy.

Postcheck artifacts:
`I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001_allen_raw_postcheck`.
`result.json` SHA256:
`252170cf3a484b23c3ea7194a49a229ebe287ffaed311f43a48fb00063dc760c`.
The exact analysis source and complete hash bindings are retained there.

## Per-donor coarse diagnostic

| Allen donor | Sections | Angle (deg) | Offset (µm) | Capture32 | Capture128 |
|---|---:|---:|---:|---:|---:|
|14452|11|45.9504|2365.713|0.36364|0.63636|
|15219|10|41.0143|5364.390|0|0|
|15336|11|40.7963|4738.463|0|0.09091|
|15439|11|37.8240|4223.426|0|0.09091|
|15447|11|45.2084|3590.637|0.18182|0.36364|
|15935|10|59.5101|4620.823|0|0.10000|
|Equal-donor macro|64|45.0506|4150.575|0.09091|0.21364|

## Gross-input check, not tuning statistics

All64 saved raw red inputs are finite and nonconstant on the fixed0–1 intensity
scale. Per-image standard deviation ranges0.008109–0.277075 and dynamic range
0.299389–0.999932. No segmentation, alternate channel, contrast normalization,
crop change or donor-specific adjustment was tested.

The following are averages of **per-image** statistics within each donor.
The saved JSON also contains minimum standard deviations/ranges and1/5/50/95/99
percentiles. These development-donor statistics must not set augmentation or
preprocessing parameters; inspect the disjoint training donors separately.

| Donor | Mean std | Mean range | Mean median | Mean p95 | Mean p99 |
|---|---:|---:|---:|---:|---:|
|14452|0.14476|0.70639|0.000046|0.39770|0.53059|
|15219|0.14599|0.67244|0.00000037|0.42219|0.52962|
|15336|0.13379|0.73372|0.00000009|0.38108|0.54476|
|15439|0.13349|0.65379|0.015715|0.36369|0.50136|
|15447|0.11569|0.65389|0.000014|0.32863|0.48455|
|15935|0.15350|0.80813|0.0000011|0.43679|0.62285|

The fixed-scale `all64_fixed_scale_inputs.png` and
`six_source_rgb_and_saved_red.png` were visually inspected. Anatomical structure
and useful red-channel contrast remain visible; one very small-tissue section
was retained. Native acquired-pixel coverage is0.864583 of the fixed12-mm canvas,
with the existing acquired-border median padding elsewhere. This excludes a
blanket constant/flat-red preprocessing failure, not every possible alignment
convention, representation mismatch or domain-transfer cause.

## Decision and limits

The previously promising synthetic retrieval captures do not transfer adequately
to this raw-image cohort. Do not promote the model or report calibrated
probabilities, electrode-region coverage, full-frame accuracy, or DeepSlice
superiority. These six donors remain development diagnostics, never training or
appearance-statistics sources. The source cohort was not certified free of all
historical benchmark exposure and is mostly near-coronal product5 fluorescence.
No alternate preprocessing or model selection was performed on these64 images.
