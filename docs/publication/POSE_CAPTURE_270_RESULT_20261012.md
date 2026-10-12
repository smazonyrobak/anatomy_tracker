# 270: bounded v9 pose-capture continuation — audited result

The trainer exited successfully after 8,000 additional optimizer batches of four independently sampled synthetic sections (32,000 accepted presentations; 35,150 attempted). It continued only this project's randomly initialized 268 pose model and AdamW state, with no external weights, features, pseudolabels, acquired labels, CAL/FINAL data, or public benchmark. The independent `training.verify_pose_capture_270` audit passed: source/config/parent/checkpoint/output hashes, disjoint physical-section IDs, all accepted draws, selection arithmetic, and 32 recomputed case readouts agree with the frozen receipt. The selected checkpoint is step 8,000, SHA-256 `d05f15cf673c688f54bd0b60eddf6de85252dbc96b2193f58f3818b2532082e4`.

| Added batches | Fresh v9 top-16 ≤1.5 mm | Fresh top-2 | Fresh oracle, all 160 | Fresh mean first-choice site error | Fresh mean normal-axis error | Old v8 top-16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27/64 | 11/64 | 37/64 | 3.113 mm | 41.3° | 31/64 |
| 2,000 | 32/64 | 14/64 | 40/64 | 3.372 mm | 39.3° | 33/64 |
| 4,000 | 38/64 | 20/64 | 44/64 | 2.909 mm | 39.1° | 44/64 |
| 8,000 | 46/64 | 28/64 | 51/64 | 3.015 mm | 34.9° | 49/64 |

The predeclared gate **fails** solely because mean first-choice error is 3.015 mm rather than ≤3.000 mm. Do not round this into a pass. The other five gate conditions pass, including top-16 capture ≥45/64 and old-panel retention ≥40/64. The gain in captured proposals is substantial; it does not establish accurate first-choice alignment, good deformation, calibrated probability, or real-slice transfer. This is a TRAIN-inner synthetic panel from eight held-gradient synthetic bases, not unseen animals or a biological validation set.

At selected step, tissue visibility exposes the main limitation: among 11 sections with <15% visible tissue, only 4/11 have *any* branch within 1.5 mm and 2/11 have one in the top 16; their mean first-choice error is 5.167 mm. For 46 sections with 15–40% visible tissue, the counts are 41/46 oracle and 38/46 top-16, mean 2.762 mm. For seven with ≥40% tissue, both counts are 6/7, mean 1.302 mm. The six >45°-from-nearest-axis sections have 4/6 top-16 captures and 3.463 mm mean first-choice error; this is too small a tail for a heavy-oblique claim. The 34.9° mean normal error remains scientifically large. Low-information partial slices may require explicit abstention, acquisition constraints, or same-brain neighboring sections; there is no basis to silently narrow the promised cutting-plane support.

The next predeclared assay, 271, asks only whether atlas *intensity* can distinguish near from far branches among the image-only top 32 at matched atlas support, using exact nine-node section thickness. A pass would justify identity-disjoint synthetic confirmation before any atlas-guided learning. A fail blocks another similar reranker and points to a different proposal/anatomical representation. Neither result promotes this pose-only checkpoint to the GUI. The eventual one-pass pose plus fragment-aware local mapping, constraints, animal-level uncertainty calibration, biological arbitrary-oblique validation, and fair DeepSlice comparison remain undone.

Frozen output: `I:/AnatomyTracker/runs/pose_capture_270_v9_continuation_001`. Protocol: `docs/publication/POSE_CAPTURE_270_PROTOCOL_20261012.md`. Independent verifier: `training/verify_pose_capture_270.py`.
