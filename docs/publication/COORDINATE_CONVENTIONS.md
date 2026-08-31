# Coordinate and transform conventions

This document is normative. Code, datasets, model metadata, figures and exports must identify their coordinate contract and version.

## Legacy coronal stereotaxic pose

- AP is micrometres from bregma.
- AP `0` is bregma.
- Anterior AP is positive; posterior AP is negative.
- L--R and D--V cutting tilt are degrees.
- In-plane rotation is a nuisance/image-orientation variable and is not a cutting tilt.
- User-selected image orientation is authoritative; automatic alignment must not silently mirror the section.

These values are a derived compatibility view only when a plane is
well-conditioned in the coronal chart. They are not the normative geometry for
the arbitrary-plane model. The frozen existing renderer defines an oblique
coronal plane in atlas voxel axes by

`AP = AP_center + tan(LR) * (ML_index - ML_center) + tan(DV) * (DV_index - DV_center)`.

Equivalently, `source/atlas_pose_runtime.py` uses the normal proportional to `[-tan(LR), 1, -tan(DV)]` in its atlas-axis ordering. This formula, rather than an informal visual description, defines tilt signs. Any change requires a contract-version increment and regression fixtures. Exact sagittal and horizontal normals are singular in this chart and must remain in full-frame/O/U/V form rather than being clipped into invented tilt values.

## QuickNII representation

QuickNII represents a plane by origin `O` and in-plane vectors `U` and `V`.
This is the normative serialized, interchange and evaluation geometry for the
arbitrary-plane model. A normalized raster point `(s,t)` maps to CCF by
`P(s,t)=O+sU+tV`, so no cutting orientation is a special case. Evaluation of
DeepSlice uses the QuickNII coordinate convention within DeepSlice's applicable
coronal scope. For an actual `W`-by-`H` raster, official QuickNII/webnutil
interoperability evaluates pixel `(x,y)` as
`P(x,y)=O+(x/W)U+(y/H)V`: `O` is the first pixel's atlas point and no half-pixel
term is added. The last pixel is at fractions `(W-1)/W` and `(H-1)/H`, while
`U` and `V` retain their full `W` and `H` spans.

The frozen `deepslice-corresponding-pixel-plane-distance-v1` diagnostic in
`training/quicknii_plane_metric.py` sampled `(x+0.5)/W,(y+0.5)/H`. It remains
unchanged solely so prior diagnostics are reproducible; it is not the future
arbitrary-plane interoperability or release-evaluation contract. New results
must use a separately versioned official `x/W,y/H` implementation.

Allen arrays use `(AP,DV,ML)` indices with AP increasing posteriorly, DV
increasing inferiorly and ML increasing toward anatomical right (PIR). QuickNII
uses right--anterior--superior coordinates in atlas voxels, so points map as
`(ML, AP_size-AP, DV_size-DV)` and vectors as `(dML,-dAP,-dDV)`. Physical
manifest points use occupied-voxel centres,
`p_um=origin_um+(index+0.5)*voxel_size_um`; therefore the inverse bridge is
`index=(p_um-origin_um)/voxel_size_um-0.5`. Plane normals are covectors: with
anisotropic spacing a physical normal maps in proportion to
`voxel_size_um*normal_um`, with its signed offset scaled by the same norm.
These conversions are named contracts; an implicit division by 25 is not
allowed.

The learned internal state is constrained: a 3-D QuickNII span centre `c`, a
right-handed orthonormal frame `R=[u,v,n]` represented by a continuous 6-D
rotation parameterization, and a positive-diagonal upper-triangular 2-by-2
in-plane basis `A`. The exact conversion is `[U V]=[u v]A` and
`O=c-(U+V)/2`. This constrained QR form has exactly the nine effective degrees
of freedom needed to round-trip every non-collinear ordered O/U/V frame,
including legitimate in-plane anisotropy and shear. Unconstrained
nine-coordinate regression is not eligible. Raster reflections remain
explicit metadata reparameterizations and never enter the proper-rotation
state. Centre/frame/basis tensors used by the renderer are in Allen array-index
coordinates; physical micrometre plane geometry is retained in the manifest
and converted explicitly before rendering.

A horizontal raster flip is an image-coordinate reparameterization, not a
physical atlas reflection. Reversing a discrete `W`-pixel raster requires

`H_W(O,U,V)=(O+((W-1)/W)U,-U,V)`, so the new pixel `x` maps to the old pixel
`W-1-x`. Vertically, `V_H(O,U,V)=(O+((H-1)/H)V,U,-V)`.

Both transforms are self-inverse when the same raster dimensions are retained.
The frozen DeepSlice comparator retains its separately versioned
continuous-boundary transform `(O+U,-U,V)` for reproducibility; that transform
must not be reused by the new arbitrary-plane renderer. Reflecting the atlas
mediolateral axis is forbidden: it changes physical hemisphere and reverses
L--R tilt. An asymmetric unilateral-landmark fixture must prove laterality,
exactly-one-flip behavior and raw/canonical plane-distance parity.

## Image coordinates

- Pixel coordinates are `(x, y)` with origin at the upper-left of the stored/displayed image unless a file format explicitly states otherwise.
- `x` increases rightward and `y` downward.
- Horizontal/vertical display flips are explicit transforms applied to the image, smart-brush points, landmarks and probe observations together.
- Brightness, contrast, opacity and zoom are display/preprocessing variables and do not alter saved anatomical coordinates.

## Dense-map directions

The current dense runtime contract is absolute pixel maps in both directions:

- `fixed_to_moving`: atlas-plane pixel to histology pixel;
- `moving_to_fixed`: histology pixel to atlas-plane pixel.

The proposed model retains both directions. Electrode marks selected on histology use `moving_to_fixed` before atlas/3-D conversion. Rendering a histology overlay in atlas space samples the histology through `fixed_to_moving`. Composition order must be covered by identity, affine, flip and known-flow fixtures before any scientific result is accepted.

## Validity and missing tissue

Visible tissue, damaged/missing tissue and atlas foreground are separate masks. A transform may be geometrically defined outside visible tissue, but correspondence metrics and scientific claims use only the declared valid domain. Missing tissue does not become a positive atlas correspondence.

## Probe coordinates

Probe entry, trajectory, depth, roll and recording sites are expressed in the same bregma-centred CCF frame used for slice poses. Angle of attack uses `0 degrees = horizontal` and `90 degrees = vertical`. Signed roll is measured in the horizontal plane relative to the bregma--lambda direction, with the sign convention fixed by regression fixtures. Atlas ontology version, voxel spacing and structure lookup hash accompany every exported mapping.
