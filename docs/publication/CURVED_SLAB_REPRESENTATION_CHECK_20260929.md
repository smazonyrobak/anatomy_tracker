# Exact coherent-subject slab: centre surface plus local director is sufficient here

The CPU-only NumPy report exited0 on all16 completed sections from
`I:/AnatomyTracker/data/joint_v6_coherent_subject_sections_001`. This is one
accepted synthetic training subject, not16 animals, model accuracy or a cohort
validation. All original coordinate arrays were authenticated against the
completed manifest; no atlas/model/GPU or live training output was accessed.
The generator permits25–100um slabs, but these16 actual thicknesses span only
33.690505–94.238913um. The extremes25/100um were not newly sampled or tested.

## Fit and physical error

Fit `C(z,x,y)=C(0,x,y)+z*d(x,y)` separately at every existing raster point.
The centre surface is the **exact saved observed centre**, not a fitted plane.
With physical PSF offsets `z` in um, ordinary least squares gives
`d=sum(z*(C(z)-C(0)))/sum(z*z)` over all nine original samples. The dimensionless
director is not forced to have unit length. Residuals compare actual saved
slab-to-CCF coordinates, never a bilinearly warped image.

RMS is the root mean squared Euclidean3D coordinate error, not per-axis RMSE.
PSF-weighted RMS uses the original normalized through-plane weights. Spatial
points receive equal weight; pooled summaries weight planes by included pixel
counts. The finite-support stratum uses saved support>0, not image intensity.

| Coordinates included | Max error, um | RMS, um | PSF-weighted RMS, um |
| --- | ---: | ---: | ---: |
| All16 planes, complete canvases | 0.163218 | 0.015330 | 0.012993 |
| Original11 raw-eligible planes, complete canvases | 0.163218 | 0.017962 | 0.015224 |
| Original11 eligible planes, finite-support pixels | 0.149462 | 0.021769 | 0.018450 |

The1um maximum-error trigger for an additional `z²` coefficient was declared
before execution. No section exceeded it, so **no quadratic term was fitted**.
The largest observed residual is0.00653 of one25um atlas voxel. This diagnoses
representation capacity only: predicting the unknown centre/director from an
image remains a separate unsolved learning problem.

Per eligible plane, complete-canvas linear coordinate errors:

| Section | Thickness, um | Maximum, um | RMS, um | PSF RMS, um |
| --- | ---: | ---: | ---: | ---: |
| 0 | 40.400784 | .033172 | .005170 | .004381 |
| 2 | 34.515093 | .021400 | .004162 | .003528 |
| 3 | 37.948844 | .032136 | .005639 | .004779 |
| 6 | 67.492988 | .105095 | .021305 | .018057 |
| 7 | 72.295578 | .098011 | .016932 | .014350 |
| 9 | 76.314627 | .111860 | .018128 | .015364 |
| 10 | 50.734933 | .048203 | .010696 | .009065 |
| 11 | 70.204092 | .082818 | .014321 | .012137 |
| 12 | 59.225583 | .067943 | .013268 | .011245 |
| 14 | 90.267137 | .147004 | .029491 | .024996 |
| 15 | 94.238913 | .163218 | .032230 | .027318 |

Sections1,4,5,8,13 remain in the full report, marked with original censor flags.
Four have no finite-support pixels; one has two. None was discarded from the
complete-canvas geometry analysis. Original per-mode eligibility is retained.

## Why the surface and director must remain distinct

Relative to the saved canonical CCF full-canvas parent-plane fit, centre-surface
perpendicular departure reaches145.730676um; full-slab departure reaches
177.721532um. These are reported separately so ordinary slab thickness is not
mislabelled curvature. Directors depart by up to6.344008degrees from that
canonical unreflected normal, with lengths0.893928–1.105434. The positive-z
direction is not the potentially reversed observed-raster cross-product normal.

Keeping the exact centre but using a constant canonical unit normal gives
all-canvas max4.762719um, RMS0.873622um and PSF RMS0.793512um. Using the fitted
planar centre as well gives max157.010853um and RMS43.221368um; eligible
finite-support RMS is60.011310um. Thus centre curvature is the dominant omitted
coordinate term here, while a local director also captures through-plane tilt
and stretch. This supports a centre3D surface plus unconstrained local director
as a minimal candidate representation for these accepted slabs, without proving
identifiability, safe topology, uncertainty calibration or subject generalization.

## Reproducible artifacts

Script: `I:/AnatomyTracker/tmp/curved_slab_representation_check_001.py`.
Output: `I:/AnatomyTracker/runs/joint_v6_curved_slab_representation_check_001`.
`analysis.json` records all16 plane results, exact offsets/weights, eligibility,
geometry departures and hashes of all32 section files. `fitted_coefficients.npz`
contains the16 fitted director fields; the exact centres remain in source data.

- Analysis JSON: `e033107df72e9f18dba734569115164aaa10fc439f9424bce780dcc843f9a785`.
- Fitted coefficients: `73d185bdc4d41f245d1bcca6e6184d2d244417478472f4d49349d7dc12e0f6f8`.
- Flat analysis source: `338f0ba3d7124d5550f471f2e15bcc534864a3ff5d054841d427567c557e0e22`.
- Input completion manifest: `dbe7a8222a58ae7de517f1a998b55ef7794f24d88cad31965fa757814497bed1`.
- Input protocol: `e86a3934df19458d76c262ccad5acd2c66811d3db26879f952d9eeacf4c2e39c`.
- Accepted subject-plan content receipt: `496669aacf08f9a5cee3aedc4fcf3a600cfe9ced106fd1358c3d4bef2cf8c7bc`.
