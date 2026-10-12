# 256: physical span supervision fixes collapse, not pose selection

The fresh-random 256 pilot completed 3,000 batches of four independently drawn eligible v7 TRAIN sections (12,000 presentations). Its draw-log SHA-256 exactly matches 254, making this a paired input-exposure comparison. The only training change was to add direct centre and physical raster-span losses to 254's normal, rigid-coordinate and reflection losses. The atlas scorer, local deformation mapper, pose refiner and uncertainty output were not trained in this stage.

The predeclared readout evaluated the batch-3,000 checkpoint on all 227 eligible, identity-disjoint synthetic DEV sections from eight deformation plans. Values are equal-plan means unless stated otherwise:

| Image-selected prediction | 247 | 254 | 256 |
| --- | ---: | ---: | ---: |
| Normal-axis error | 54.66° | 50.97° | **55.91°** |
| Surviving-tissue rigid coordinate error | 4.247 mm | 4.410 mm | **4.773 mm** |
| All 128 branches, truth-selected coordinate error | — | 2.290 mm | **2.156 mm** |
| Predicted minor physical FOV ≤1.2 mm, cases | 161/227 | 33/227 | **0/227** |

The all-branch oracle is a diagnostic ceiling, **not** a deployable prediction: only 7/227 sections have any 256 branch within 1 mm, and the image-selected branch is 2.617 mm worse than the oracle on average. Best of the top four normal-mass anchors under either reflection is still 3.212 mm. The selected centre error is 2.660 mm and reflection accuracy 61.7%. Normal cross-entropy over the final 500 TRAIN batches is 4.126, barely below the uniform 64-anchor value of 4.159; the measured DEV normal error worsened relative to both earlier heads. The size term taught realistic field of view but did not solve where the section belongs or which candidate to trust.

The frozen continuation gate required selected rigid error ≤4.0 mm, normal-axis error ≤51°, and at most 10 collapsed-FOV cases. Only the last condition passed. Thus **do not scale the unchanged 256 training loss**, use this checkpoint for the GUI, calibrate its probabilities, or run the public DeepSlice comparison. The result does not imply arbitrary-plane localization is physically impossible: the prior model-free exact-plane synthetic assay separated ±1.5/3 mm offsets and ±15° tilts, while this one-pass model has seen only 12,000 training presentations. It shows the present direct 64-anchor factorization/supervision is not yet an adequate route to a shippable single-section estimate.

The next intervention should directly address geometric pose capture and selection, rather than another scalar ranking head or more size supervision. A bounded matched test of a spatially grounded pose representation, with clean and artifact-heavy independently sampled sections, can distinguish a learnability bottleneck from artifact-domain difficulty before allocating large exposure. Keep all brain-intersecting planes in the target distribution; reserve acquisition constraints, nonrigid fitting feedback and probability calibration until the pose proposals contain sufficiently accurate candidates without truth-assisted choice.

The evaluator exited successfully. Independent read-back verified 227 unique physical-section IDs across eight synthetic plans, exact row and summary SHA-256s, matched 254/256 draw logs, the narrow-FOV and ≤1 mm oracle counts, and the failed gate. The frozen TRAIN and DEV outputs are `I:/AnatomyTracker/runs/pose_factor_256_train_001` and `I:/AnatomyTracker/runs/pose_factor_256_readout_001`; their `completed.json` receipts bind sources, protocols, checkpoints and result hashes. This was synthetic DEV only, with no CAL/FINAL biological animals, calibrated uncertainty or public benchmark.
