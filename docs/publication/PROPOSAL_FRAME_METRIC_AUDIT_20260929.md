# Is the 43.7° error a convention bug?

The targeted independent audit reproduces **43.7264978348°** on all 640 frozen
curriculum003 development rows. No angle-unit, antipodal-normal, fixed
axis/sign, saved-truth conversion, or O/U/V cross-product discrepancy was found.
This narrows the diagnosis; it does not certify the entire learning pipeline.
Active005 was not accessed, and no GPU inference or training was run.

## Independent numerical routes

The audit imports no project geometry helper. NumPy independently decodes each
12-value state into physical O/U/V using its two rotation vectors, positive
in-plane scales and shear. Unit normals are `cross(U,V)/norm(cross(U,V))`.
An independent second route converts the original frozen row JSON's QuickNII
U/V vectors directly to physical AP/DV/ML and takes their cross product. These
small metadata reads verify all 640 row identities; no image, warp or dense
coordinate replay was repeated.

| Check | Result |
| --- | ---: |
| Independent plane-normal mean | 43.7264978348° / 0.7631713576 radians |
| Maximum difference from saved per-row error | 6.10e-13° |
| Maximum difference using original QuickNII U/V truth | 2.11e-13° |
| `acos(abs(dot))` versus `atan2(norm(cross),abs(dot))` | ≤5.78e-13° |
| Signed-normal angle folded with `min(a,pi-a)` | ≤4.45e-16 radians difference |
| All 48 shared signed axis permutations | ≤4.37e-13° change |
| Best single fixed one-sided signed axis convention | 42.6957°, still poor |
| 100 shuffled prediction/row pairings, mean | 57.5065° |
| One-row prediction shift, mean | 59.0852° |

The one-sided search applies one fixed transform to every prediction, not a
truth-selected transform per row. Its best result is an ML reflection (or its
antipodal equivalent); no axis permutation/sign convention produces near-zero
error. The independent signed-direction mean is instead 52.0638°, demonstrating that
the 43.7° endpoint already correctly allows the plane normal's `n/-n` ambiguity.
Prediction-shuffle controls are descriptive, not biological significance tests.

## Source-contract inspection

- State layout is `[center3, first_rotation_vector3, second_rotation_vector3,
  log_scale_x, log_scale_y, shear]`. Frame axes are **columns** `[u,v,n]`.
- Physical edges are `U=exp(log_scale_x)*u` and
  `V=exp(log_scale_y)*(shear*u+v)`; `O=center-(U+V)/2`. Positive scale/shear
  composition preserves the cross-product normal.
- QuickNII vectors `[ML,AP,DV]` become Allen vectors `[-AP,-DV,ML]`, then use
  25µm spacing. Origins use the corresponding point conversion. Saved truth
  is the canonical effective physical plane; raster reflection is separate.
- The finite renderer uses `O+(x/W)*U+(y/H)*V`, then adds physical PSF offsets
  along the normal. Physical voxel centers obey `origin+(index+0.5)*spacing`.
  PyTorch volume dimensions are AP/DV/ML, while `grid_sample` coordinates are
  ML/DV/AP; that reversal is intentional. `align_corners=True` matches its
  `(size-1)` grid normalization.
- The metric takes the physical frame's third column for prediction and truth,
  clips the absolute dot product, computes `acos`, then converts radians once.
  Independent predicted O/U/V normals match stored catalogue normals within
  2.23e-16 per coordinate.

Relevant sources: `arbitrary_plane_geometry.py`,
`arbitrary_plane_full_frame_primitives.py`,
`arbitrary_plane_training_data_v6.py`, and frozen003 `experiment_source.py`.

One error **in our supplementary symmetry note**, not the model, was found:
its physical midline labels omitted the half voxel. Array reflections
`455-index` / `456-index` correspond to 5700 / 5712.5µm, not 5687.5 / 5700µm.
The note and derived metadata are explicitly corrected; voxel differences and
normal-angle diagnostics are unchanged. A translation cannot alter these
normal-angle errors.

## Next discriminating experiment

Overfit a fixed set of eight identifiable, non-duplicate training slices using
the actual image encoder plus complete 98,304-cell proposal head and corrected
FP32 computation. Keep images, labels, masks and objective fixed. Require
memorization and geometry approaching the already-established catalogue
quantization limit before another large run. Failure would direct investigation
toward expressivity, gradients, optimizer behavior, or label/input logic;
success would instead prioritize data diversity and generalization. This is an
optimization diagnostic, not validation or evidence of a deployable model.

## Receipts

CPU script and result are under `I:/AnatomyTracker/tmp/`:

- `proposal_frame_metric_audit_20260929.py`: SHA-256
  `970242a8fd65417a91d69826718885698aaf598225c3e885d80bce889979e10c`.
- `proposal_frame_metric_audit_20260929.json`: SHA-256
  `21744a2a0b82ecc42cdaf26c934aa501b53e49ed9881294892dc9616cdc88dbc`.

The JSON includes all 48 conventions and exact input receipts. Frozen003 final
row-array SHA-256 is
`da976905e2a75d3cac22ef60e661701fa492f38dc50440c0fc2ead9feff96ec8`.
