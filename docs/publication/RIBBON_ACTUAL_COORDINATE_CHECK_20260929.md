# Ribbon constructor: frozen-coordinate check, 2026-09-29

**Result:** no coordinate-composition defect found on the 16 frozen coherent-subject slabs. The constructor preserves the existing fitted linear ribbons without clipping. This is a CPU geometry check, not model training, optimization success, biological validation, or a benchmark.

Inputs are `I:/AnatomyTracker/data/joint_v6_coherent_subject_sections_001` and the saved directors in `I:/AnatomyTracker/runs/joint_v6_curved_slab_representation_check_001/fitted_coefficients.npz`. All 16 planes remain included: 11 raw-support-eligible, 5 censored; 9 are horizontally reflected. They come from **one** accepted synthetic subject, at actual thicknesses 33.69–94.24 µm. The earlier [representation fit](CURVED_SLAB_REPRESENTATION_CHECK_20260929.md) describes their limitations.

## Construction and independent comparison

The saved canonical O/U/V fit determines a proper frame R and upper-triangular physical basis A. Spatially unreflect the observed centre and fitted director before computing `r=Rᵀ(C0−P)` and `d=RᵀD−e3`. Construct `P+Rr+z(n+Rd)` with maximum derivative norm 0.35, then reflect only the output raster x-index. Do not reflect CCF components, the physical positive-z director, or anatomy. Unreflected centres equal the saved canonical centres exactly.

Independent NumPy corner derivatives use `[∂s(r+zd), ∂t(r+zd)] A⁻¹` plus axial derivative `d`, at both physical slab extrema. Their maximum Frobenius norms agree with the primitive within 1e-12 for every plane. The index/corner review found the physical-basis inverse and four-corner axes consistent with the bilinear-raster, linear-z contract.

| Full-canvas quantity | All 16 | Eligible 11 |
|---|---:|---:|
| Constructor versus saved linear fit, max µm | 1.22229e-10 | 1.22229e-10 |
| Constructor versus exact slab, RMS µm | 0.01533013334 | 0.01796235038 |
| Constructor versus exact slab, PSF-weighted RMS µm | 0.01299309019 | 0.01522406790 |
| Constructor versus exact slab, max µm | 0.1632183836 | 0.1632183836 |
| Prelimit derivative-norm range | 0.110868–0.152314 | 0.124091–0.152314 |
| Applied deformation scale | exactly 1 throughout | exactly 1 throughout |

The approximation errors remain those of the original linear-through-slab fit; the constructor adds only floating-point roundoff. Maximum removed affine coefficient is 1.19024e-10 µm; maximum postprojection uniform affine moment is 9.42437e-15 µm. All measured unscaled corner Jacobian determinants are positive (minimum 0.853056). This is not a claim about a subsequently composed section-processing map, outside-raster extrapolation, or unseen targets. The active clipping branch was not exercised because these actual targets require no clipping.

## One actual-data backward pass

Section 0, CPU float32, actual exact-coordinate MSE: 2.80617e-5 µm². All gradients are finite and nonzero: state L2 norm 7.55427, surface-residual 8.08035e-5, director-delta 4.48281e-5. Maximum float32 coordinate difference from the saved float64 linear fit is 0.00359498 µm. No optimizer or model was used; differing gradient scales are not an optimization-quality assertion.

## Reproduction and immutable references

Executed once, exit 0: `I:/AtlasJointProject/envs/npixel_analysis/python.exe I:/AnatomyTracker/tmp/ribbon_geometry_check_001.py`. Output: `I:/AnatomyTracker/runs/joint_v6_ribbon_geometry_check_001/analysis.json`. It contains per-plane results/IDs, all 32 input-section file hashes verified against completion, and the source hashes. Geometry source bytes were unchanged across execution. Only this result note is committed; no primitive was edited.

| Artifact | SHA-256 |
|---|---|
| Flat check script | `894139691f1e5f641392b6fd44d3041b49ff667374d4f0e2befbc329289f079a` |
| Analysis JSON | `aff81737a692bcdca5b5f9ef8c35b65094700019f5629456bfecf235adb51449` |
| `arbitrary_plane_ribbon_v6.py` | `d556936140462747f67e4881d3629d435218d9e22696ab99fa6856794abd3f1f` |
| `arbitrary_plane_full_frame_primitives.py` | `cf1f0edc31c96b1baa877224268f8e1297b82d4d7e689fbb24df6be4d1473817` |
| `arbitrary_plane_geometry.py` | `432f2c58fff0c5df033072bee42ee49e6d5cf39391edd7891b15260318753030` |
| Input `completed.json` | `dbe7a8222a58ae7de517f1a998b55ef7794f24d88cad31965fa757814497bed1` |
| Saved fitted coefficients | `73d185bdc4d41f245d1bcca6e6184d2d244417478472f4d49349d7dc12e0f6f8` |
