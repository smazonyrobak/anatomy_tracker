# Coupled pose-volume 181: TRAIN-only gate failed

The [frozen protocol](COUPLED_POSE_VOLUME_181_TRAIN_PROTOCOL_20261011.md) tested whether a 16-branch, 27-action-per-branch rigid pose volume and finite-thickness atlas-image matching could select a better arbitrary plane than the frozen direct pose prior. It was a mechanism test, not joint model training. Both arms saw the same 128 independently generated eligible TRAIN sections (64 each of v3 and v4), the same candidate states and atlas supports. The control permuted atlas intensity within each candidate's own support. No DEV, real animal truth, final animal, or public benchmark was used.

| Frozen population result | v3 (64) | v4 (64) |
| --- | ---: | ---: |
| Direct / intact / permuted-null mean tissue-site error, mm | 2.607 / 2.833 / 2.732 | 2.646 / 2.750 / 3.199 |
| Direct / intact ≤1.5-mm capture | 40.63% / 39.06% | 37.50% / 34.38% |
| Original-16 / intact-shortlist truth-best ≤0.5-mm availability | 14.06% / 31.25% | 17.19% / 29.69% |
| Direct / intact mean five-frame-point error, mm | 6.541 / 6.970 | 7.151 / 7.464 |
| Intact selected-action mean original-support retention | 98.79% | 98.63% |
| Paired intact-plus-null inference p95, seconds/section | 0.478 | 0.479 |

The enlarged local search passes the *availability* condition but fails blind selection: intact mean tissue error increases by 0.226 mm in v3 and 0.104 mm in v4, ≤1.5-mm capture falls in both, and the five-point result regresses. The intact selector is also 0.100 mm worse than its matched intensity-null in v3, though 0.449 mm better in v4. The predeclared continuation gate is **false in both appearances**; supported pairwise selection and runtime pass, but the accuracy and 99% mean-support-retention requirements do not.

An independent read-only audit verified the completion receipt, all four output hashes, committed assay/protocol/source and parent bindings, exact one eligible section per base and appearance, 128 unique TRAIN physical IDs/seeds/plane geometries, row-to-draw and candidate-index/error bindings, shortlist and selected-action calculations, and every summary metric and gate. A post-exit decomposition of those same frozen rows attributes +0.249/+0.133 mm to changed branch choice; within the chosen branch, the selected local action recovers only 0.023/0.028 mm. The selector changed branch in 49/64 sections in each appearance. The all-27-action *oracle* within its selected branch could recover about 0.310/0.299 mm, but that oracle is not an inference result. The intact arm's shortlisted local oracle could recover only 0.108/0.112 mm; coarse shortlist truncation and fine ranking both lose potential. The original 16-branch truth oracle was already about 1.706/1.668 mm better than the direct choice, so branch ranking remains the larger measured bottleneck.

**Decision:** do not promote this rigid intensity-correlation selector to DEV, the GUI, or pose-head feedback training. A favorable exact-plane comparison in assay 180 and increased truth-best availability here do not make a 432-way blind selector accurate. This failure says nothing definitive about a jointly learned pose-and-local-warp model or about whether informative physical steep-oblique sections are identifiable. The next model work should couple pose mass and tissue mapping to one image-conditioned fitting objective, with a frozen TRAIN-only comparison against direct pose and matched atlas-information controls; animal-level validation and probability calibration remain prerequisites for deployment.

Frozen output: `I:/AnatomyTracker/runs/coupled_pose_volume_181_train_001`. SHA-256: `completed.json` `4e361faa5841c3b4895325f7666767622122d88f640a75a9071e79ae3ee5e591`; `summary.json` `4b1e5a37d598a1da51b9ad62069b5a5bf234076157f6657b603e0efe9816c51d`.
