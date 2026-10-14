# 305 fluorescence adaptation leaves exact-pose local mapping effectively unchanged

The [predeclared TRAIN-inner audit](JOINT_SPATIAL_305_EXACT_MAP_RETENTION_PROTOCOL_20261014.md) compared the matched batch-600 control and fluorescence-adapted checkpoints on the same frozen 64-case panel. Each checkpoint received the saved image and exact post-artifact state/reflection, finite-thickness offsets and weights, atlas, and 512 sampled valid-pixel CCF targets. The pose refiner was not called. The local map used the existing 305 joint-readout semantics at 96x96. No model was updated and no DEV, calibration, final, or public-benchmark data were read.

| Frozen panel | Zero warp | Control exact map | Adapted exact map | Adapted - control |
| --- | ---: | ---: | ---: | ---: |
| All 64 cases, mean | 0.117431 mm | 0.111300 mm | 0.111646 mm | +0.000346 mm |
| Strong-warp top 16, mean | 0.184684 mm | 0.167177 mm | 0.167358 mm | +0.000181 mm |

The control and adapted maps improved the full-panel zero-warp mean by 0.006131 and 0.005785 mm (5.22% and 4.93%). On the predeclared strong-warp stratum they improved it by 0.017507 and 0.017326 mm (9.48% and 9.38%). Neither arm reached the older joint-readout 10% strong-warp clause. Adaptation was better on 18 cases and worse on 46.

Under the deliberately direction-only harm rule, adaptation technically harmed the full-panel and strong-warp means. The magnitude is only **0.346 micrometres overall (0.311%)** and **0.181 micrometres on strong warps (0.108%)**, so this is not a material local-map regression on this frozen panel. An independent parameter comparison found all warp, atlas encoder, pair, pose-refiner, reliability, and uncertainty tensors byte-value unchanged between arms; only the deliberately optimized encoder, lateral, and pose modules differed. The tiny output change therefore comes through the adapted shared image features, not changed map-head weights.

Independent post-run aggregation reproduced all 64 rows, the 16-case stratum, and every summary mean. Frozen output: `I:/AnatomyTracker/runs/joint_spatial_305_exact_map_retention_001`. Source commit `9eef6c699e1ebf2982673066866d4cc5ad29a6b7`; panel SHA-256 `53ca593e97fdcd02c5c8db66172f45662fbed2c98017e16ac7dbebc89194611f`; control/adapt checkpoint SHA-256 `646fa4166d0dea57d996b7bf485778e4b45e4a894a37db759902de06fd6c07bd` / `03fc26319140bb6c8742ddad571e35caaf5ab833829b953594dfeef65f216b39`; rows SHA-256 `00f726a5c889729235c17cef3889d32e65c6c53382ae51684786cc578de20da6`; summary SHA-256 `60df067d7f7bcdad2bfe2f8811f034f0cffe5e21479876a29ac7353dc0c10561`.
