# Sixteen coherent-subject training sections

Prepared driver: `training/prepare_joint_v6_coherent_subject_sections.py`.
Output: `I:/AnatomyTracker/data/joint_v6_coherent_subject_sections_001`.
Preparation is CPU-only and is not launched by this source change.

Use the completed, accepted plan in `joint_v6_coherent_subject_pilot_001`,
whose process exited successfully before its artifacts were read. Pin the
completed check hash, original plan/receipt/context files and recomputed plan
content receipt `496669aacf08f9a5cee3aedc4fcf3a600cfe9ced106fd1358c3d4bef2cf8c7bc`.
The subject plan remains unchanged: root seed 2026092907, standard nonidentity
3D deformation, accepted amplitude 62.5 um, one synthetic subject. Authenticate
once on load; do not repeat the expensive plan sampler or independent replay.

Section root seed 2026092908 deterministically draws 16 arbitrary planes, each
with independent geometry/appearance RNG domains. All retain the plan's train,
animal, synthetic-animal, specimen and experiment IDs, with new section IDs.
For each plane sample its own 25–100 um thickness and normalized nine-sample
PSF. Uniform antipodal normals, uniform roll and conservative subject-box slab
offsets preserve support for all orientations; no low-tissue draw is rejected.
The 16 draws are not a coverage certificate. Physical raster spans follow each
subject-box projection: they are not implicitly 12 mm or 125 um/pixel.

Observed raster -> identity **section-processing** pullback -> recorded
horizontal reflection -> subject plane/slab -> shared subject-to-CCF map.
The identity processing map is deliberately distinct from the nonzero total
anatomical warp. Store the exact canonical anatomy centre grid and its fitted
OUV/3D residual, observed centre/slab coordinates and total-map fit/residual,
subject slab coordinates, reflection, PSF and RNG provenance. Full-canvas fits
are independent of brush masks; residuals may have out-of-plane components.

Decode the pinned v6 Allen intensity/support volume and original annotation.
Its normalized intensity differs from the pilot's v2 scalar preprocessing;
both atlas bindings remain explicit and reference the same raw Allen files.
Recompute nearest PSF labels (ties-to-even), central brain mask, modal purity,
central-label support and nearest occupancy from the new mapped coordinates.
Keep these separate from the trilinearly sampled finite-PSF support channel.
Never copy an old row's anatomy labels or claim a brush is automatic segmentation.

Each plane produces three paired observations: raw synthetic background,
exact-black exterior and imperfect brush. They share the same continuous
render, reflection, gain/gamma/inversion, noise and background before masking.
Brush masking occurs after noise, making removed pixels exactly zero. Retain
all 48 observations, including empty/low-support cases. Record finite and
visible support masses and the >=64-pixel information censor; this is not yet
a finalized coarse-pose supervision rule.

Geometry is saved once per plane with its three observations using the existing
JSON/NPZ writer; plan snapshots, source hashes, exact IDs and artifact hashes
are retained. Expected output is roughly 0.1 GB. The 16 CPU mappings query
about 1.47 million physical points in total, without caching a new 3D volume.

**Not compatible with legacy total-2D-SVF packs.** The data declare
`target_kind=coherent_subject_3d_coordinate_targets_only` and
`legacy_total_2d_svf_pack_compatible=false`; they contain no `truth_state`,
`truth_catalogue_index`, `truth_stationary_velocity_yx_px` or total-warp
`truth_pullback_map_yx_px`. `catalogue_cell_truth` remains null until a
subject-canonical gauge/label rule is explicitly established. A zero 2D
processing map cannot supervise total anatomical deformation as zero.

This is coherent synthetic training-data preparation from one atlas and one
subject realization, not independent biological animals, held-out validation,
model performance, a benchmark, calibrated uncertainty or a shipping result.

## Completed CPU preparation

Source commit `3d48a91aa5900aada4aa8e60f575994e79f34281`; session 29655 / PID
22044 exited 0. Output access waited until exit. Completion took 331.21 seconds
and produced all 16 unique section IDs / 48 observations, one shared animal ID,
all split `train`, with zero rejected draws. Five planes are support-censored
in each mode; eleven planes / 33 observations pass the recorded support test.
Section artifacts total 115,252,066 bytes.

The completed manifest is
`I:/AnatomyTracker/data/joint_v6_coherent_subject_sections_001/completed.json`,
SHA-256 `dbe7a8222a58ae7de517f1a998b55ef7794f24d88cad31965fa757814497bed1`.
It records each section's JSON/NPZ paths and hashes. Existing
`_read_raw_artifact(output_directory, section_record["artifacts"])` restores the
exact arrays without regenerating the plan or assuming legacy SVF labels.

For the eleven support-eligible planes, full-canvas canonical anatomy-fit
out-of-plane RMS ranges 22.49–45.25 um (mean 29.62 um). This describes stored
3D target geometry, not registration error, tissue-only deformation statistics
or accuracy of a trained model. No development/held-out cohort was created.
