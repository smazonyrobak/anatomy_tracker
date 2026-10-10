# 135: added grain is visible but does not close the measured texture gap

The frozen TRAIN-only QC drew 256 eligible v4 and 256 eligible v5 physical sections from **independent random streams** over the 64 TRAIN deformation bases. It logged 576 attempts, 576 distinct attempted IDs and 512 distinct accepted IDs; no source plane was deliberately recolored or paired across arms. Each section retained the original arbitrary-angle pose, finite-thickness and observed-pixel CCF targets. Source, protocol, frozen-119 reference, rows, attempts, summary, image-grid and output hashes passed the post-exit audit. Every checked raw/exact-black exterior pixel was exactly zero (8,873,475 v4 and 8,357,524 v5 pixels); dense targets remained finite and invalid coordinates zero.

The main frozen-119 comparison is the raw/exact-black stratum at 192 pixels. Its threshold-based tissue proxy found measurable interiors in 144/153 v4 and 125/136 v5 images. The arms are **unpaired**, so their small differences include random plane, anatomy, artifact and exposure variation.

| Interior metric, mean | Acquired TRAIN in 119 | v4 raw/black | v5 raw/black |
| --- | ---: | ---: | ---: |
| 1-pixel high-pass RMS / contrast | 0.09350 | 0.08077 | 0.08063 |
| 3-pixel high-pass RMS / contrast | 0.20236 | 0.19054 | 0.18733 |
| 10th–90th percentile contrast | 0.55802 | 0.17957 | 0.20685 |

The v5 grain raised absolute 1-pixel high-pass RMS in these unpaired samples from 0.01351 to 0.01571, but raised contrast too; normalized fine texture did **not** move toward the acquired reference. The actual-generator grid visibly shows grain in some v5 tissue, yet both versions still look like transformed atlas images rather than physical histology. The grid uses metadata-selected, **different** planes, and display-stretched input intensities; it is a visual check, not a matched pixel comparison or model prediction.

**Decision:** keep v5 as an exploratory generator variant, but do not use it in model training merely because it looks grainier. Do not tune its amplitude against the same QC scalar or claim improved real transfer. The next appearance step should study real TRAIN tissue texture and staining jointly with contrast and spatial scale, then use a fresh unpaired generator QC and animal-separated weak-real guard. This does not relax the arbitrary-plane objective; physical steep-oblique expert truth is still absent. No checkpoint, GUI model, uncertainty calibration or public benchmark was changed.

Frozen output and grid: `I:/AnatomyTracker/runs/v5_tissue_grain_qc_135`; completed-receipt SHA-256 `4567c4d29c0d06d25af764e0756df3c2fb234b87541394325f426ff3639faada`; PNG SHA-256 `2ebe55f17212218d36dec50e9e4fed628142441ee6da7106da856d8d85700336`.
