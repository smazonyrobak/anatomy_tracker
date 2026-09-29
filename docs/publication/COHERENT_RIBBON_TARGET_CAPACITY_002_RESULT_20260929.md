# Coherent640 target-capacity screen

**The fitted target fields fit the current numerical ribbon limits. This is not
decoder learnability, trained accuracy, biological validation or qualification.**

Section generation68817 exited0 after1343.483s, retaining all640 arbitrary-plane
sections and1,920 paired observations from12 synthetic subject maps. There were
zero rejected draws. All work is on `I:`; no real animal is represented by these
synthetic subject IDs.

| Split | Subjects | Sections | Raw censored | Exact-black censored | Imperfect-brush censored |
|---|---:|---:|---:|---:|---:|
|Train|8|512|201|201|212|
|Development|4|128|37|37|40|

Each section has all three observation modes. Censored observations remain in
the cohort and this descriptive capacity screen; no threshold was relaxed.

## Post-exit capacity result

Audit36825 exited0 and authenticated every section metadata/array artifact.
It used the original FP64 coordinates, undid horizontal reflection spatially
only, and recovered the canonical fitted frame. In that frame it measured
`r=R.T@(C-P)` and `d=R.T@(D-n)`, where
`D=sum_s(w_s*z_s*(slab_s-C))/sum_s(w_s*z_s*z_s)` is the exact-centre-anchored
weighted least-squares world director. All640 sections/1,920 observations are
reported, including full-canvas and visible-support summaries by subject, mode
and censor status.

| Measurement, maximum over all sections | Result | Native limit |
|---|---:|---:|
|Absolute local surface component|150.933268µm|200µm before affine projection|
|Absolute local director-delta component|0.14884720|0.2|
|Physical derivative Frobenius bound|0.17806434|0.35|
|Affine-gauge coefficient magnitude|1.92635e-10µm|Numerical residual|
|Linear-slab reconstruction3D point error|0.257013µm|Descriptive approximation error|
|Per-section PSF-weighted slab RMS error|0.0357121µm|Descriptive approximation error|

No component reaches either cap and no joint derivative rescaling activates:
all rescale factors equal1. Spatial-unflip and stored fitted-plane reconstruction
errors are exactly0. The all-section mean PSF-weighted slab RMS error is0.011223µm.
Full-canvas normal-to-plane residual RMS ranges5.272–39.687µm, median23.696µm;
therefore a flat plane still omits genuine generated out-of-plane deformation.

The surface cap precedes affine projection; generally a single representative's
cap exceedance would not prove that no bounded affine-equivalent prefield exists.
Here no exceedance occurs. The clipped/limited closure reproduces the same slab
fit error up to numerical precision. The screen reuses the native derivative/
gauge implementation; it is not an independent mathematical implementation or
a guarantee that the lower-resolution neural decoder can learn these fields.

## Frozen provenance

- Cohort: `I:/AnatomyTracker/data/joint_v6_coherent_subject_cohort_sections_002`.
  Completion SHA256:
  `ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2`.
- Audit: `I:/AnatomyTracker/runs/joint_v6_coherent_ribbon_target_capacity_002`.
  Summary SHA256:
  `4d3568ffccb2323d11a1f3bba551d51c1ef44c2fc9e7a121bbd53cd0507e6405`.
- Saved audit source SHA256:
  `f83b9fd0e1a2e29e3827f0fa3b92a7f5411972b1fa3dcd25fb50ef3fbca821f8`.
  This pinned source was launched before its commit in `3d37b71`; the saved
  source is the exact execution reference.
- `sections.jsonl` SHA256:
  `0dd05edbd9d3b4bb3d68965a1641b2f2bbc46896cb52734b85ecf3f223d01ac2`.
- `observations.jsonl` SHA256:
  `0f095c714afe9740b5ba6a489e55fa37f1fa0d5b5ae1026f825d6466b2c219cc`.

Proceed to the separately predeclared native conditional learning experiment;
do not widen the caps on the basis of this cohort or claim a working model from
target representability alone. No renderer, model checkpoint, training or active
training-output inspection was used by this capacity audit.
