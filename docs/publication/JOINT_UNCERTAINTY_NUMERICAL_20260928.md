# Joint uncertainty numerical primitives — 2026-09-28

`training/arbitrary_plane_joint_uncertainty.py` adds functions for a local
35-coordinate distribution: nine physical frame residuals and 26 smooth
velocity coefficients. A bilinearly interpolated 4x4 vector grid loses its six
uniform-canvas affine modes before SVD orthonormalization. The basis is built
once per raster size and retained by its caller. It uses the current decoder's
canvas gauge, not a different tissue-weighted gauge.

The covariance is `diag(scale**2) + factor @ factor.T`, with rank four shared
between pose and deformation. The likelihood computes exact selected-coordinate
Gaussian marginals; omitted targets do not become zero targets. All-missing
examples require exclusion from loss averaging. Frame residuals invert
`compose_full_frame_state` in the open rotation-angle-below-pi chart. Sampling
flags excursions outside that chart, and does not silently clip velocities,
reject folds, or claim calibration. Velocity projection reports residual energy
outside the 26-mode basis; that omitted variation is not covered by this model.

This is a numerical component ready for a future head on the recurrent context.
It has not been wired into the trained model and is not evidence of anatomical
accuracy, electrode confidence, or calibration.

## Reproducible numerical experiment

One CPU experiment used seed `2026092804`, double precision, an 8x8 raster,
and 30,000 joint draws. It checked inverse frame composition, the basis gauge,
coefficient recovery, masked likelihood against a dense Gaussian, and empirical
joint covariance. It generated numerical mathematical fixtures, not training data.

| Quantity | Result |
| --- | ---: |
| Maximum inverse-composition error | 4.4054e-13 |
| Zero-residual gradient finite | true |
| Maximum basis orthonormality error | 2.6645e-15 |
| Maximum affine moment | 2.4032e-16 |
| Maximum coefficient recovery error | 1.7764e-15 |
| In-basis omitted velocity energy | 1.5867e-29 |
| Maximum marginal NLL difference from dense Gaussian | 7.1054e-15 |
| Sample covariance relative Frobenius error | 0.015493 |
| Pose/velocity cross-covariance relative error | 0.018820 |

Script: `I:\AnatomyTracker\tmp\joint_uncertainty_numerical_001.py`.
Result: `I:\AnatomyTracker\tmp\joint_uncertainty_numerical_001.json`.
SHA-256 receipts:

- Module: `5aba5971d986cd495c9c7cde29329b6f5229e09befb4a7ebe3d1476e572ce73a`
- Script: `659992f7d627bc63560f0b2c9dc122c48742133c0b626aecf478d393766849fd`
- Result: `52311291661e5ce3c15fd48db9722ffe3d66535f4d09300951ba9bdc6456fc5c`

## Preserve raster-representation uncertainty

`canonicalize_representation_raster` uses output-to-input affine-grid mappings.
The recurrent renderer first flips the atlas raster for each representation,
then applies the shared cell pullback warp. Thus an observed electrode point
maps as `base_atlas_raster(A_representation(W_cell(point)))`: warp first, selected
reflection second. Pixel reflections are exactly `x -> width-1-x` and
`y -> height-1-y`; the subsequent QuickNII coordinates use `x/width, y/height`.

Draw the physical cell, its conditional raster representation, and then the
correlated local frame/velocity residual. Use the same representation for all
marks within the sampled section. A small shared-weight uncertainty head can
operate on each representation's final recurrent context. Averaging reflected
positions or pooling away this discrete ambiguity can put a false trajectory
near the midline. The global unrefined probability mass must remain unresolved.

The scientific motivation and future animal-level calibration procedure remain
in [JOINT_CONSTRAINT_UNCERTAINTY_UPDATE.md](JOINT_CONSTRAINT_UNCERTAINTY_UPDATE.md).
