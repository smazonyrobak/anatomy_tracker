# Raw Allen proposal diagnostic geometry

`training/evaluate_joint_v6_allen_raw.py` is a flat diagnostic for a **completed**
new joint-model experiment. Set its `CHECKPOINT` constant before running it as a
module; do not run against a live writer. It selects only the six development
donors/64 images of the frozen `allen_real_development_20260928` cohort. No public
benchmark data or legacy learned features are used.

Inputs are the entire unsegmented, acquired red-channel JPEG field, divided by
255, with no tissue-dependent crop, rotation, normalization or automatic mask.
A fixed acquisition-centred 12-mm field is resampled to 96² with Gaussian
antialiasing and bilinear interpolation. Pixels outside the acquired field use
the median acquired border intensity, not an inferred tissue exterior. Outline
and outline availability are both zero. A fixed 12-mm field can clip the outer
acquisition context and pad shorter dimensions; it does not promise that the
entire acquisition fits. Every exact source-pixel transform and padding value is
saved, along with the actual input pixels and original image hashes.

The downloaded pyramid has factor 32: source full-resolution pixel centres are
`(downloaded_pixel + 0.5) * 32 - 0.5`. The full-size/downloaded-size ratio is not
used because pyramid edge pixels are truncated. The model renderer samples
`(x/96, y/96)`, not half-pixel-shifted normalized positions, so model pixel 48
maps exactly to the acquisition centre. The recorded full-pixel matrix has
diagonal `12000/(96*resolution_um_per_px)` and centre-preserving translation.

Reference geometry follows the official
[Allen image-to-reference affine formula](https://api.brain-map.org/doc/SectionImage.html#image_to_reference-instance_method):
`tsv` maps full image pixels to the section volume, volume z is
`section_number * section_thickness`, and `tvr` maps that volume to PIR micrometres.
PIR is the model's increasing AP/DV/ML axis order. A `(12.5,12.5,12.5)`-µm shift
then expresses these coordinates in this implementation's physical voxel-centre
convention: `physical = (Allen_index + 0.5)*25`, matching the existing
QuickNII-to-index-to-physical training path. This is a coordinate-convention
conversion, not an empirical registration correction. The composed physical map
is saved per image; its first two columns define the reference normal by their
cross product. Ground-truth geometry never enters the image or model inputs.

The diagnostic reports MAP-cell antipodal normal-angle error and signed-offset
error relative to the catalogue support origin, averaging sections within each
donor and then donors equally. Offset comparison aligns normal signs first;
for nonparallel planes, its value depends on the explicitly recorded origin.
Raw full-catalogue log probabilities and per-section predictions are retained.

These are coarse affine-reference diagnostics, not anatomical landmark errors,
dense deformation labels, expert consensus, calibrated confidence, or proof of
arbitrary-plane histology generalization. They cannot identify in-plane roll,
translation, scale/shear, nonrigid distortion or electrode localization quality.
Allen's coronal fluorescence cohort and the historical exposure limitations in
`ALLEN_REAL_DEVELOPMENT_20260928.md` remain material. A single affine plane does
not recover the true section's local warping or reference-alignment uncertainty.

CPU-only geometry preflight on all 64 acquired development sections found a
maximum scalar-versus-composed-affine discrepancy of `3.64e-12` µm and centre
plane discrepancy of `1.88e-12` µm. One independent live official API check,
section `112514723` at full pixel `(19991.5,14991.5)`, exactly matched the local
PIR result `(1844.4849692899988,5964.7918055725,6982.1984862624995)` µm.
Reference plane normals range from 1.42° to 7.83° off coronal. The fixed physical
grid samples acquired pixels at 86.46% of positions; the remainder is padding.
Source syntax parsed successfully. No checkpoint, active experiment output or
GPU was accessed for this preflight; model inference remains unexecuted.
