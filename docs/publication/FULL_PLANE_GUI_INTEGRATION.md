# Full-plane GUI geometry foundation

2026-09-28. This is integration groundwork, not a released model or an accuracy
claim. The new model is not enabled in the desktop application.

`SliceSession.atlas_ouv_ap_dv_ml_um` stores absolute physical Allen CCF O/U/V
vectors; `atlas_raster_shape_h_w` stores their fixed raster. Pixel `(x,y)` maps
to `O + (x/W)U + (y/H)V`. The training atlas uses 25-um AP/DV/ML voxels with
voxel-face origin zero: voxel index zero has physical centre 12.5 um. Therefore
the GUI index conversion is `physical_um / 25 - 0.5`, matching the training
renderer. These are not QuickNII RAS coordinates.

One session-aware mapping now drives probe-coordinate creation/recomputation,
atlas-plane sampling and 3D plane corners. Existing `SliceAtlasTransform2D`
remains the display-to-fixed-raster mapping. Neither full-plane rendering nor
probe mapping requires an outline. With no O/U/V, the legacy paths are retained.
Full-plane session archives use version 2 so old applications reject them;
legacy-only archives remain version 1, and the updated loader accepts both.

Coronal controls and legacy model start/result-install paths cannot replace a
full-plane result. Geometry, mask and landmark edits invalidate joint mappings
and derived coordinates. O/U/V remains only an atlas reference chart, preserving
the meaning of existing atlas landmarks. Pending or stale results cannot be
exported as a current joint alignment.

The remaining annotation callers were inspected: the main histology/atlas
`ImagePanel` has no region-hover lookup; exported channel regions sample actual
3D coordinates in `_sample_region`; anatomy-dialog hover/pin labels use a
separate probe-aligned vertical plane reconstructed from mapped CCF sites, not
the current coronal controls. Its template and annotation share the same grid.
Other coronal samplers belong to the guarded legacy optimizer/warp paths.
`from __future__ import annotations` supports the geometry helpers appearing
before `SliceSession`.

A small numerical check found zero legacy mapping discrepancy. A follow-up
review caught a missing half-voxel shift in the original GUI conversion (the
initial check compared physical points, not voxel indices). After correction,
the GUI matches `physical_um_to_allen_index_points` within 3.56e-15 voxels and
the differentiable finite-thickness renderer on a linear volume within 7.11e-14
intensity units. O/U/V corners use the same corrected conversion. Syntax and
diff checks passed. No GUI was launched or packaged for these checks.

Before release: install the actual trained joint runtime and exact image
preprocessing/dense-map directions atomically with O/U/V, raster dimensions,
provenance and registration state; wire optional constraints; integrate and
validate posterior trajectory/site samples and calibration; verify session
round-trips and real GUI workflows. This foundation does not yet display
uncertainty or provide a full-plane manual-pose editor. The atlas preview samples
the central plane, not a finite-thickness PSF. Build the replacement on I: and
update the desktop shortcut only after qualification; do not modify the old
C:/F: installation or depend on its trained models.
