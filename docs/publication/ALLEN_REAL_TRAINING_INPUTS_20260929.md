# Raw Allen training-only inputs and appearance comparison

Preparation exited0 from `affaeacb817db0ff5e860f8c536e6cb7509ab2fe`:
`I:/AnatomyTracker/data/allen_real_training_inputs_20260929` contains all256
previously downloaded images from58 **training** donors as float32[256,1,96,96],
exact row/animal/specimen/experiment/section identities, raw hashes, original
affines and complete pixel transforms. The six development donors are explicitly
excluded before any image read. No new download, model inference, segmentation,
alternate channel, learned label or training occurred.

The contract is unchanged: acquired red/255, acquisition-centred12-mm field,
Gaussian antialiasing, bilinear resampling and acquired-border median padding.
Outline availability is false. Upstream Allen affines are provenance and coverage
metadata, not certified dense anatomy or synthetic catalogue targets.

## Frozen bindings and training statistics

Saved output hashes and the source snapshot were authenticated after exit.
`summary.json` SHA256:
`4f8484bbd2ab5719971d9862f3f5ea87ea3d1c9de76468a936af67bcc65da9fd`.

| File | SHA256 |
|---|---|
|raw_model_input.npy|`b4c0bc908ca4b61a4000bbaa02b140e440c3b4da42069b6b793e435c64f38379`|
|image_geometry.jsonl|`9a718b2fb2791584e1d225123facbe1c02c96cfa701a9e6f3fc68f17a565d218`|
|input_statistics.npz|`5fcdc34f13b1dd7f09ead393fd7cdeb4a8ad7dcf6740b2c4352ff9e6f4807b86`|
|preparation_source.py|`f17022e971c9054b80e4bd1d61b2994495e425948b8003147d378ad86070db33`|

The summary stores per-donor counts, exact section IDs, std/range min/mean/max,
mean per-image1/5/50/95/99-percentiles and affine-normal ranges. Equal-donor mean
image std is0.1298886; corresponding quantiles are
`[0,0,0.0009110,0.3548625,0.5156587]`. Per-image std spans0.028391–0.199774 and
dynamic range0.415138–0.998139. These are full-canvas statistics, including
acquired/padded background, not automatically segmented tissue statistics.

Upstream normal angles to the AP axis span0.486985–11.627585degrees, and maximum
pairwise antipodal normal separation is13.240406degrees. Affine normals are
largely per-experiment observations, not independent section poses. These data
cannot provide full arbitrary-plane coverage; preserve synthetic coverage.

## Training-only synthetic appearance comparison

A flat CPU analysis read authenticated frozen5120 **training** rows and the first
eight saved generated training observations from completed image-key001. It did
not read development images, evaluate model predictions, choose alternate
channels, or tune a transformation grid. First-eight examples are illustrative,
not a census of all generated appearances. Fixed-scale contact sheet and complete
statistics:
`I:/AnatomyTracker/runs/joint_v6_train_only_appearance_comparison_001`.
Result SHA256:
`eeb6629d704a11c7bce4ce62527f3f103741252ff738b9243d7a34db3c5ffb5f`.

| Training inputs | Count | Mean std | Mean median | Mean p95 | Mean p99 | Exact-zero fraction |
|---|---:|---:|---:|---:|---:|---:|
|Real, equal-donor macro|256 /58 donors|0.12989|0.00091|0.35486|0.51566|0.43406|
|Synthetic absent brush, positive weight|1608|0.20013|0.48323|0.80392|0.89976|0.02452|
|Synthetic accurate brush, positive weight|1616|0.24434|0.01869|0.65063|0.83187|0.69507|
|Synthetic imperfect brush, positive weight|1619|0.24969|0.01909|0.66833|0.85387|0.69347|

Synthetic rows have arbitrary-plane tissue occupancy, unlike the real near-
coronal cohort; image-mean quantiles are not matched anatomical measurements.
Nevertheless, the large absent-brush/background mismatch is concrete: real
unsegmented acquisitions often have black/near-black exterior, whereas synthetic
absent-brush rows usually have substantial background intensity. Masked synthetic
black exteriors generally arrive with **outline availability1**, unlike raw
acquisitions. The model may exploit that association; this is a hypothesis, not
a demonstrated cause of prediction failure.

The actual first32000 generated observations used tissue gain0.60001–1.39998,
gamma0.60003–1.59996, background mean0.0000044–0.799986, whole-raster additive
noise std0.00500–0.05000 and50.15% tissue inversion. Thus moderate darkening is
already within the augmentation support, and gain/gamma alone cannot create the
missing exact-zero mass or fix contrast relationships that are not monotonic.

## Bounded next hypothesis, not an approved training change

The more direct minimal control is **outline/availability dropout** on existing
accurately and imperfectly masked synthetic inputs: with probability0.5 set both
boundary and availability to0, leaving image pixels, pose, PSF, deformation,
supervision weights and original source mode unchanged. Apply it to generated
and frozen training rows, not development evaluation. This exposes the existing
black-exterior images without making a brush mandatory or requiring automatic
segmentation. Continue a whole checkpoint; do not merge independently trained
encoders. A matched A/B protocol and launch remain the parent's decision.

The train-only source census confirms the exact existing association:

- Frozen5120: absent1706, accurate1707, imperfect1707; all3414 masked rows have
  availability1, and3235 have positive point weight. All1706 absent rows have
  boundary and availability exactly0. Twenty masked rows have an all-zero
  boundary yet availability1: a zero boundary is not itself missing-metadata
  evidence, so dropout should operate on the explicit availability/mode.
- Frozen first4000-step presentations: absent10632, accurate10676,
  imperfect10692. Generated32000 schedule: absent10662, accurate10756,
  imperfect10582. The first eight stored generated channel tensors agree with
  the mode rule; later generated channels were not retrospectively re-rendered.
- No dropout was applied. Exact counts and source binding are retained in
  `dropout_train_counts.json` in the comparison directory.

If testing a zero-preserving tone control, a concrete engineering range is
`x'=g*x**gamma`, `g~Uniform(.55,.85)`,
`log(gamma)~Uniform(log(1.1),log(1.6))`; a representative centre is approximately
`g=.70,gamma=1.4`. Positive gamma and zero bias preserve exact zeros. Do not add
noise outside an explicitly synthetic known support when the selected exterior
is required to remain black. This is a broad, training-only intensity-scale
hypothesis, not a fitted acquisition model or uniquely inferred parameter set.

The dropout control retains raw/background, exact-black and imperfect-brush
cases while exposing black/near-black synthetic exteriors with no outline.
Do not require or infer a real tissue mask. Merely applying gain/gamma to the old
positive raw backgrounds does not test that missing condition. Keep geometry/labels/PSF and
full arbitrary-plane coverage unchanged, and separate this appearance control
from structural/domain changes in interpretation. No such source edit or
training run is included in this result.
