# 254: direct full-pose signal failed the independent synthetic DEV gate

The fresh-random direct 64-anchor model completed 3,000 batches of four independently sampled eligible v7 TRAIN sections: 12,000 accepted presentations. Its draw-log hash exactly matches 251's, so the two direct-head runs used the same physical sections and appearances. The 254 loss additionally supervised physical pose and raster reflection. The map, atlas-fit ranker and refiner received no gradients and are not usable components of this checkpoint. No prior weights, CAL/FINAL animals, public benchmark, or probability calibration were used.

The frozen batch-3,000 checkpoint was evaluated once on all 227 eligible sections from the eight synthetic DEV plans in panel 248. Subject-equal means are below; these plans are synthetic variations of one atlas, not biological animals.

| Readout | 247 image-only | 251 CE-only direct head | 254 full-pose direct head |
|---|---:|---:|---:|
| Antipodal plane-normal error | 54.66° | 52.78° | 50.97° |
| Image-selected rigid tissue-coordinate error | 4.247 mm | not trained/evaluated | 4.410 mm |
| Selected error if only raster reflection were chosen by truth | not in this readout | not trained/evaluated | 3.692 mm |
| Centre error | 2.783 mm | not trained/evaluated | 2.739 mm |
| Predicted minor field of view ≤1.2 mm | 161/227 | not trained/evaluated | 33/227 |

The true minor field of view is never ≤1.2 mm (0/227). The 254 mean absolute major/minor field-of-view error is 7.473 mm, and its reflection classification accuracy is 68.3%; these remain poor. Its last-500-batch TRAIN normal CE is 4.0643 nats, versus 4.1589 for uniform mass and the preset pass limit of 3.659. The preset DEV limits were ≤42.78° and ≤3.75 mm. **All three gates failed.** Coordinate error improved over 247 in only two of eight synthetic plans. Of the 254 cases with ≤15° axis error, nine still have ≥2 mm selected coordinate error.

Thus direct pose supervision reduces the severe span-collapse symptom but does not produce reliable full-pose localization or selection. The 0.718-mm gap between selected and reflection-oracle errors shows an additional reflection-selection problem, but even oracle reflection leaves several millimetres of error. The next diagnostic should separate candidate capture from candidate selection across all 64×2 branches, before another training change. Do not scale or deploy this loss unchanged, and do not claim arbitrary-plane inference impossible from this failed model.

The readout exited successfully. An independent read-back matched all 227 unique physical IDs from eight synthetic plans and independently recomputed row, summary and checkpoint SHA-256s. Its completion record is `I:/AnatomyTracker/runs/pose_direct_full_pose_254_readout_001/completed.json`; the fixed checkpoint hash is `4d516c5bd422830eac7f652f684db85fc46a4213890470828695918fb51c2fc9`, the readout rows hash is `85f1ac7a7f0e472bfd3936dcb01773da21641f715909cde0a4bfbec578f2ddba`, and the summary hash is `76fdbe1900182f183a0a2635ea051ab0bb6341ad709e5e774c838a4060a45a52`.
