# Frozen tensor pack for conditional joint-refinement learning

Completed once on CPU in 107.6 seconds using
`training/prepare_joint_v6_local_refinement_pack.py`, commit `199d516`.
No model/GPU inference, generation, benchmark access, or source-cache changes.
Every selected source row was authenticated once by the existing frozen loader;
minibatch training can now index tensors without repeating file verification.

Directory: `I:/AnatomyTracker/data/joint_v6_local_refinement_frozen_001`.
`pack.json` records exact source commit/file hashes, trusted cache receipts,
prepared-input receipts, output hashes, field shapes and dtypes.

| Partition file | Rows | Original cache/prepared indices | Bytes | Pose-positive | Dense-positive | Pose × dense eligible |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `training.pt` | 2,048 | 3,072–5,119 | 1,523,211,135 | 1,863 | 1,864 | 1,863 |
| `internal_development.pt` | 256 | 384–639 | 190,450,819 | 231 | 231 | 231 |

All rows were retained, including zero-weight rows. One training row has positive
dense weight but zero pose weight: preserve the existing joint loss's product
of pose and dense eligibility, rather than supervising it accidentally. Original
and prepared indices match exactly, but both explicit mappings are saved.
Training reflection counts are 1,025 identity / 1,023 horizontal; development
counts are 129 / 127. These rows represent 128 / 16 organizational synthetic
groups from one atlas, not independent biological subjects.

The 3,072 training / 384 development pose-only anchors are intentionally not
copied. Their frozen contract assigns zero dense-deformation supervision, so
identity-G1 status is not permission to add zero-deformation targets silently.

## Fields and conventions

- `channels`: original float32 `[N,3,96,96]`; `image` and `outline` are
  storage-sharing `[N,1,96,96]` views; `outline_available` is `[N]`.
- `truth_state`: canonical effective physical frame, float64 `[N,12]`;
  `truth_catalogue_index`, `prepared_row_index`, `frozen_cache_row_index`: `[N]`.
- `truth_stationary_velocity_yx_px`, `truth_pullback_map_yx_px`: float64
  `[N,2,96,96]`, preserving the frozen arrays without another quantization.
  These targets have float32 generator ancestry; float64 storage is not a claim
  of additional physical precision. The source-to-fixed pullback is `exp(-v)`;
  targets are already transformed into the observed reflected raster.
- `target_ccf_coordinates_ap_dv_ml_um_float64`: `[N,3,96,96]`, original dense
  AP-DV-ML truth and units. Preserve its valid/abstention masks in any metric.
- Original correspondence weight, valid/abstention/deformation masks and tissue
  truth are `[N,1,96,96]`. `deformation_weight` exactly equals original weight
  times `(deformation_valid & correspondence_valid & ~abstention)`.
- `pose_supervision_weight`, `dense_deformation_supervision_weight`, and the
  alias `retrieval_supervision_weight` retain the frozen row decisions.
- `axial_offsets_um` / `axial_weights`: float64 `[N,9]`; exact per-row
  `finite_psf_contracts` are retained. Use global PSF normalization and zero
  padding, not per-pixel renormalization.
- `reflection_representation_index`: int64 `[N]`;
  `reflection_representation_affine_xy_float64`: `[N,3,3]`, original homogeneous
  **pixel** affine (`x -> 95-x` for horizontal reflection). This is not the
  normalized 2×3 affine expected by sampling. The bound catalogue has two states;
  catalogue alternatives, not truth reflection, belong in model inputs.
- `records` retain every lineage ID, frozen cache record with metadata/array
  hashes, row/source receipts, PSF/gauge references and support contracts.
  Dense annotation IDs are not copied because geometric losses do not use them;
  their authenticated source arrays remain untouched in the original cache.

## Decoder-range diagnostic

On combined pose/dense-eligible rows, supported per-row maximum absolute SVF
fractions have training y/x medians 0.00495 / 0.00500, 99th percentiles
0.01393 / 0.01353, and maxima 0.02209 / 0.01881. Development maxima are
0.01461 / 0.01428. Full-canvas maximum gradient Frobenius norms have training
median / 99th percentile / maximum 0.09367 / 0.20809 / 0.26049; development
maximum is 0.22317. No eligible row exceeds the decoder's 0.35 gradient limit.
No target-amplitude cap problem is evident against its 0.08 raw fraction scale.
This is not a test of network expressivity. The frozen targets' certification
uses seven scaling-and-squaring steps and a uniform full-canvas affine-free
gauge, matching the decoder gauge; use the same integration convention for the
initial local control.

This pack supports explicitly **teacher-forced local learning**, not an honest
global-retrieval evaluation, calibrated uncertainty, or biological validation.

SHA-256: training `5a909727f8c62f5eb79f75eeec8cbd39d804295c74d72b736fa185f9e6fe3cea`;
development `5200f29fc3fef3581b4a7533a99debf52e40578793214d116f319b50d0bb596a`.
