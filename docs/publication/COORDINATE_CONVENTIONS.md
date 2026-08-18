# Coordinate and transform conventions

This document is normative. Code, datasets, model metadata, figures and exports must identify their coordinate contract and version.

## Public stereotaxic pose

- AP is micrometres from bregma.
- AP `0` is bregma.
- Anterior AP is positive; posterior AP is negative.
- L--R and D--V cutting tilt are degrees.
- In-plane rotation is a nuisance/image-orientation variable and is not a cutting tilt.
- User-selected image orientation is authoritative; automatic alignment must not silently mirror the section.

The existing renderer defines an oblique plane in atlas voxel axes by

`AP = AP_center + tan(LR) * (ML_index - ML_center) + tan(DV) * (DV_index - DV_center)`.

Equivalently, `source/atlas_pose_runtime.py` uses the normal proportional to `[-tan(LR), 1, -tan(DV)]` in its atlas-axis ordering. This formula, rather than an informal visual description, defines tilt signs. Any change requires a contract-version increment and regression fixtures.

## QuickNII representation

QuickNII represents a plane by origin `O` and in-plane vectors `U` and `V`. The joint model may use O/U/V internally, but conversion to and from public AP/L--R/D--V values must use one tested implementation. Evaluation of DeepSlice uses the official corresponding-pixel plane-distance definition and coordinate convention.

A horizontal raster flip is an image-coordinate reparameterization, not a physical atlas reflection. For normalized horizontal pixel coordinate `s`, the same physical plane is represented after a flip by

`H(O, U, V) = (O + U, -U, V)`, so `P(H(E), s, t) = P(E, 1 - s, t)`.

`H` is its own inverse. The corrected DeepSlice comparator therefore preserves raw images and raw-frame ground truth, applies one deterministic raster flip into DeepSlice's expected A-to-P view, runs all official angle/order processing in that view, and applies `H` to the final O/U/V result before raw-frame scoring. Reflecting the atlas mediolateral axis is forbidden: it changes physical hemisphere and reverses L--R tilt. An asymmetric unilateral-landmark fixture must prove laterality, exactly-one-flip behavior and raw/canonical plane-distance parity.

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
