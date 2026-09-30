# Frozen Allen TRAIN-image acquisition — 2026-09-30

`I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001` exited with code 0:
263,754 full-canvas 192-pixel images from 1,885 Allen donors, with specimen,
experiment, section and donor IDs retained. No invalid-geometry sections were
excluded. The images have weak upstream Allen affine coordinates, **not** blinded
expert reference alignments or precise deformation truth.

After exit, `load_reserved_real_train()` verified all 7,541 frozen file bindings,
unique section IDs, donor receipts, array shapes and TRAIN roles. The downloaded
donor IDs exactly equal the 1,885 preassigned TRAIN donors: none overlaps the 223
development, 105 calibration, 113 final-reserved or five benchmark-excluded donor
reservations. No non-TRAIN images were downloaded in this acquisition.

Frozen acquisition manifest SHA-256:
`04a7d1cee45064919f58b2cc24b0f3295bd202500a78c55764c625f6f94f0132`.
Preassigned donor-reservation JSONL SHA-256:
`48352c9c7dbd3489ae17135a0e3ee90d7de5139a36a7ab18bdcfb5eaf033270b`.

This makes a large, provenance-preserved real **training** source available for
the same standalone model lineage. It is not held-out validation, calibration or
a reason to claim anatomical accuracy from those weak labels alone.

The affine geometry is overwhelmingly near-coronal: median plane-normal offset
from the AP axis is 4.91 degrees (10th–90th percentile 1.98–8.03 degrees), with
no section above 30 degrees; recorded section thickness is 100 µm throughout.
Thus this large real source cannot itself establish arbitrary-plane competence.
Synthetic full-plane training and independently annotated oblique real material
remain necessary.
