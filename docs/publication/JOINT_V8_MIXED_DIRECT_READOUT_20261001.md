# Mixed direct-pose continuation — frozen internal readout (2026-10-01)

The standalone v8 whole model continued from the exactly replayed million-plane checkpoint at step 161,000. It completed 20,000 more optimizer updates with 160,000 eligible arbitrary-plane synthetic presentations and 160,000 weakly registered real TRAIN-image presentations. The real sampler drew one of 1,885 TRAIN donors uniformly per update; no old application model weights or pseudolabels were used. The final step-181,000 checkpoint is `I:/AnatomyTracker/runs/joint_v8_mixed_direct_001/joint_step_181000.pt`, SHA-256 `176eed9a5a64eff130afd775de09fc43e6ad9b7302d8d9473f6cc9292b5a355b`. The completed counters, final log row, 20,000 training rows, 176,269 synthetic draw rows, and 160,000 real draw rows agree.

The matched internal readout used the same 64 sections from four held-out synthetic maps in three background conditions, and 64 real sections from six separate development donors. Values below are subject/donor-equal means. Error is the mean physical distance at five full-canvas points; lower is better. The “oracle” chooses the best of 16 predicted mode/reflection branches using the reference and is **not an inference result**.

| Development input | Selected before → after | Oracle before → after | Selected plane-normal error before → after |
| --- | ---: | ---: | ---: |
| Synthetic, raw background | 6.890 → 6.761 mm | 0.879 → 0.859 mm | 29.52° → 29.72° |
| Synthetic, exact-black exterior | 6.419 → 6.487 mm | 0.845 → 0.835 mm | 29.07° → 28.33° |
| Synthetic, imperfect brush | 5.808 → 5.465 mm | 1.145 → 1.027 mm | 28.40° → 26.55° |
| Real, weak Allen affine | 12.597 → 0.464 mm | 3.059 → 0.455 mm | 16.96° → 2.92° |

The result repairs the severe weak-real-label regression of synthetic-only training while largely preserving synthetic candidate coverage. It does **not** solve candidate selection on arbitrary planes: selected synthetic errors remain roughly 5.5–6.8 mm despite submillimetre oracle errors in two of the three conditions. The next stage therefore trains all 16 anatomy-fitting branches to feed physical fitting evidence back into pose prediction and candidate ranking, with continued real TRAIN pose retention.

The real references are upstream Allen affines, not blinded expert anatomical alignments. Synthetic map IDs are not independent biological animals. These numbers do not establish deployable anatomical accuracy, calibrated uncertainty, or a DeepSlice comparison. The complete matched readout is `I:/AnatomyTracker/runs/joint_v8_mixed_direct_dev_002`: all 512 raw prediction digests and the row/protocol digests verified. An earlier readout attempt in `_001` stopped after its parent half because this checkpoint omitted a redundant `modes` metadata field; that partial tree was preserved and was not used for the paired result.
