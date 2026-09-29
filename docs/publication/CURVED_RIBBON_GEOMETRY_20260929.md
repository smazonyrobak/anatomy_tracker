# Pose-separated finite-thickness curved ribbon

`training/arbitrary_plane_ribbon_v6.py` implements three differentiable geometry
functions. It does not load weights, change the running experiment, infer a pose,
or establish learned performance. The separate opt-in recurrent integration
must be trained before these fields become useful model predictions.

The [bounded actual-coordinate check](RIBBON_ACTUAL_COORDINATE_CHECK_20260929.md)
reproduced all16 frozen fitted slabs to within1.23e-10um in FP64. An independent
NumPy vertex calculation matched derivative bounds0.111–0.152; all target scales
were1. One actual-plane FP32 backward pass had finite pose/field gradients.
This checks coordinate arithmetic, not learned registration. The active clipping
branch and GPU native model remain untested by that check.

## Representation and target evidence

For the existing proper frame R, in-plane basis A, and base plane P(s,t), use

`X(s,t,z) = P(s,t) + R r(s,t) + z [n + R b(s,t)]`, with `n=R e3`.

The six fields are local centre displacement r in micrometres and local director
increment b in micrometres per micrometre. They are canonical B,3,H,W arrays,
bilinearly interpolated over the actual pixel chart s=x/W, t=y/H. Depth z is the
original physical section-depth coordinate, not estimated atlas-normal depth.
The director is not normalized: it must retain through-plane stretch and shear.

The [actual one-subject representation check](CURVED_SLAB_REPRESENTATION_CHECK_20260929.md)
found that exact centre surfaces plus local linear directors fit16 frozen slabs
within0.164um maximum error. Thicknesses measured were33.69–94.24um. Director
norms ranged0.894–1.105, with tilt up to6.34deg from the fitted plane normal.
These measurements support this first approximation, not arbitrary-cohort
accuracy or accuracy at unmeasured thicknesses. Retain exact nonlinear targets
and measure representation error again for larger anatomical variation.

Only r is projected against the fixed uniform full-raster basis1,s,t. Its nine
affine displacement coefficients belong to the full OUV pose. This projection
is independent of image content, tissue visibility and smart-brush mode. It fixes
an affine coordinate gauge, not statistical identifiability of every unknown.
In particular, separate unconstrained tangential anatomy and section-processing
maps can compensate each other. Keep processing identity in the first coherent
subject stage; never reuse an old2D-SVF target as a3D-ribbon target.

## Coupled finite-domain deformation bound

Before rotating into CCF, the displacement derivative relative to the physical
undeformed slab is

`J = [r_s + z b_s, r_t + z b_t, b] blockdiag(A^-1, 1)`.

For continuous bilinear fields its entries are separately affine in s,t,z.
Convexity of the Frobenius norm therefore bounds each raster cell by its four
corners at both depth extrema. The implementation evaluates these corners,
using forward cell differences and exact physical basis conversion, not sampled
centred differences or tissue-only Jacobians. The depth interval includes all
supplied PSF offsets and zero.

Let L be the maximum over all those vertices for one hypothesis. After the
affine projection, multiply **both** r and b by the same scalar
`alpha=min(1,0.35/L)`. This preserves their coupled slab mapping and the affine
gauge. For the defined continuous interpolant in exact arithmetic, the residual
Lipschitz constant is at most0.35. On the convex base parallelepiped, point
separation is consequently between0.65 and1.35 times its original value.
The canonical mapping is injective and orientation-preserving there.

This argument requires a proper orthonormal frame, invertible positive in-plane
basis, identical interpolation in the bound and evaluator, and one global scalar
per hypothesis. It gives a continuous piecewise-smooth bi-Lipschitz mapping,
not a globally C1 diffeomorphism. Floating-point reconstruction and gradients
still need actual-data checking. It says nothing about extrapolation beyond
the finite canvas/slab or any subsequently composed processing transform.

## Rendering, reflection and uncertainty

Pass the complete B,S,H,W,3 physical CCF grid to
`render_finite_thickness_coordinate_grid`, retaining normalized positive PSF
weights over original z. Do not render a planar image and then warp its pixels.
No tissue-dependent PSF normalization or extra Jacobian multiplier belongs in
this pullback intensity model. Keep known PSF thickness initially: learning
both unrestricted director scale and unknown thickness creates a scale ambiguity.

For the identity-processing stage, horizontal reflection permutes the grid's
width axis after canonical construction. It does not flip CCF components,
reflect brain anatomy, or negate the director. Evaluate intrinsic topology in
canonical coordinates, before this discrete raster orientation change. With
future processing, composition must be observed pullback -> finite reflection
-> canonical ribbon -> CCF sampling, not a second conjugation of the map.

Pose and ribbon must be updated from the same recurrent correlation evidence,
rerendered after every update and final scoring. Preserve distinct reflection
hypotheses rather than averaging their geometries. Coordinate evidence must
describe the rendered curved surface, not a stale planar proxy.

No ribbon covariance is yet trained or calibrated. The old35D pose/SVF covariance
does not describe these fields. A thin slice weakly constrains the director;
later uncertainty must jointly sample pose, discrete reflection, deformation and
trajectory, then be calibrated on unseen biological animals before any region
probability or credible-volume claim.

## Primary literature and scope of inference

[Osechinskiy and Kruggel](https://pmc.ncbi.nlm.nih.gov/articles/PMC3335496/)
combine global alignment with three-dimensional warping of a two-dimensional
histological section. This supports representing normal displacement; it does
not establish our ribbon director, topology bound or neural architecture.
[NeSVoR](https://pmc.ncbi.nlm.nih.gov/articles/PMC10287191/) explicitly models
continuous slice acquisition and PSF effects. Its image-noise uncertainty is
not evidence of calibrated registration or electrode-location uncertainty here.
