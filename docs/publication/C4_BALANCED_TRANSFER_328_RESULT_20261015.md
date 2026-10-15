# 328: weak-real-loss strength affects sparse mean, but not enough to advance C4

This 1,000-batch from-scratch arm changed **only** the acquired weak-loss multiplier, 0.3→0.1, relative to 327. The batch-0 model tensors were exactly equal across arms (all 113 state tensors), and the full accepted/rejected synthetic and acquired draw log was byte-identical (SHA-256 `973527e08948184ea4967bd5fbf9011fcb6b07c1ce731291bb6d32e7a01f5e69`). Each arm still used fresh independently random cuts *within that arm*; cross-arm replay is a matched scientific comparison. Both used the same donor-disjoint TRAIN-inner and synthetic panels. No previous weights, DEV/CAL/FINAL animals or public benchmark were used.

| Batch-1,000 readout | 327: weak loss ×0.3 | 328: weak loss ×0.1 |
| --- | ---: | ---: |
| Inner first-choice mapped mean (128) | 3.500 mm | 3.488 mm |
| Inner cases over 5 mm | 14 | 14 |
| Steep-oblique mapped mean (32) | 3.901 mm | 3.869 mm |
| Sparse mapped mean (32) | **6.833 mm** | **6.085 mm** |
| Sparse cases over 5 mm | 24 | 21 |
| Coronal first-choice AP family (10 donors) | 10 | 10 |
| Sagittal first-choice ML family (8 donors) | 8 | 8 |

All nine [predeclared ablation clauses](C4_BALANCED_TRANSFER_328_PROTOCOL_20261015.md) passed. The paired sparse *mean* gain is 0.748 mm, but it is heterogeneous: only 15/32 sections improved, 17 worsened, and the paired median gain is −0.047 mm. An exploratory 10,000-resample case bootstrap gives a 95% interval of **−0.076 to 1.681 mm** for the mean gain. The three largest improvements account for most of it; omitting them leaves a 0.148-mm mean gain. Thus this specific frozen panel shows sensitivity to weak-loss weight, **not** a robust general improvement or proof that real gradients caused all sparse failures. Its sparse mean remains 0.529 mm worse than 321's batch-1,000 result on the same panel. Do not perform a series of finer weight sweeps from this small result.

The local map remains ineffective: mapped-minus-rigid mean error in 328 is −0.00002 mm inner, +0.00063 mm steep, and −0.00035 mm sparse. Correct family labels on acquired TRAIN donors are not exact physical pose or local correspondence accuracy. This model is uncalibrated, fails absolute sparse quality, and is not a GUI replacement or a DeepSlice competitor. The efficient next design step is an architecture/data change that addresses sparse-plane proposal capture and learns a nontrivial tissue map, with early acquired and synthetic gates—not more exposure of this 6.23M C4 objective merely because the matched relative gate passed.

Post-exit audit matched every output SHA-256 receipt, source/protocol/model and 327 binding hashes, 0/1,000 readouts, 1,000 consecutive training rows, all 210 final case rows, raw-row means and family counts. Run: `I:/AnatomyTracker/runs/c4_joint_balanced_328_001`; completion SHA-256 `776f3c06be54e05afb4f9e71e4046cf3042fbecfb48fc5977c99edaba1d3da9f`. `matched_ablation_gate_passed=true`; `model_promoted=false`, `calibrated=false`.
