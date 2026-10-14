# 309 acquired fluorescence transfer: four-TRAIN-animal audit

A read-only diagnostic applied the identical raw `max(red, green)`/p99 input policy, four zero auxiliary channels and 192-pixel grid to all 240 valid Lupori sections from four **TRAIN** animals. It compared 309 batch 0 with batch 20,000 on the same images. The human VisuAlign planes were read only for post-inference, approximate five-site comparison; they are not measured physical truth. `CC2B`, `CC4B` and `CC4A` entered 309 gradients. `AL1A` did not enter those gradients but has been used in development, so it is a TRAIN-inner guard, not untouched validation.

| Animal | Sections | Batch-0 first-choice gap | Batch-20,000 gap | Batch-20,000 median unsigned normal gap |
| --- | ---: | ---: | ---: | ---: |
| `AL1A` (309 guard) | 52 | 1.239 mm | 1.161 mm | 4.79° |
| `CC2B` (309 gradient) | 54 | 0.362 mm | 0.194 mm | 1.30° |
| `CC4B` (309 gradient) | 66 | 0.880 mm | 0.383 mm | 2.86° |
| `CC4A` (309 gradient) | 68 | 0.817 mm | 0.305 mm | 2.08° |

The large gains on gradient animals must not be called transfer. The small `AL1A` gain (0.078 mm) supports only retention on one near-coronal acquired TRAIN animal; its top-16 ≤1.5-mm count moved 49→48/52. This narrows the process decision: a generic failure to ingest this fluorescence style is **not** the evident dominant 309 problem, whereas blind arbitrary-plane synthetic first choice remains ≈2.5 mm and sparse cuts much worse. Do not spend the next large run on fluorescence-only augmentation, nor infer real arbitrary-oblique, electrode-localization or calibrated-probability accuracy from these approximate coronal references. A genuinely donor/animal-separated acquired gate and measured physical obliques remain necessary before GUI replacement.

The process exited normally. Post-exit read-back matched all three completion-manifest hashes, all 480 raw rows and 240 identical paired image/section hashes, and the per-animal raw-row means. Frozen I:-only output: `I:/AnatomyTracker/runs/first_choice_309_lupori_train_audit_001`; source commit `ce2eecb`, source SHA-256 `699d1b7cff4b16f16004daa3350b261724e67da78cdc94c444a76a9914cff9d3`; completion SHA-256 `74c08e24ebeb4954ad2998b7549640930569e6f03bde1432216534fc24b362fa`.
