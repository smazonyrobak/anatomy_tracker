# One-shot 011: correct-pose mapper apprenticeship

The 4,000-batch continuation from frozen 010 completed. It updated the local mapping and atlas/image comparison path while freezing the global pose and fitted-ranking heads. The source, checkpoint, row-level predictions and hashes are recorded in `I:/AnatomyTracker/runs/one_shot_true_pose_mapper_011/` and `I:/AnatomyTracker/runs/one_shot_true_pose_mapper_011_development_eval/`. Evaluation completed with 640 rows; independent SHA-256 checks matched both evaluated checkpoints, the evaluator, `rows.jsonl` and `summary.json` to the frozen receipt.

On the same 185 eligible synthetic development images (eight synthetic identities derived from one atlas), identity-equal mean error in micrometres was:

| Metric | Step 0 | Step 4000 |
| --- | ---: | ---: |
| Correct pose, rigid mapping | 139.4 | 139.4 |
| Correct pose, learned mapping at deployment 256 px | 186.4 | 123.8 |
| Correct pose, learned mapping at training 96 px | 197.7 | 115.6 |
| Predicted top-prior pose, learned mapping | 2903.3 | 2915.9 |
| Fitted-selected pose, learned mapping | 2998.3 | 2921.3 |

The correct-pose mapper improved on 183/185 images and beat the rigid mapping on 129/185 at 256 px. It is now useful *when the pose is supplied correctly*. End-to-end fitted selection improved on only 79/185 images, however, and its mean 2.92 mm error still fails to beat the unchanged top-prior baseline (2.90 mm). The fitted choice flipped on 40/185 cases even though the fitted-ranking head was frozen, because its upstream features changed; this comparison does not isolate mapping from ranking. The fitted-selected 90th percentile fell from 5.99 to 5.67 mm, but the worst case remained 12.19 mm.

On 64 real development images from six donors, the five-point **rigid pose** error against existing weak Allen alignments was unchanged at 571.3 µm for top prior, and 618.8 to 621.6 µm for fitted selection. This does not measure real deformation quality. The correct-pose synthetic mapping uses oracle pose/reflection, not an end-to-end prediction; the eight synthetic IDs are not independent biological animals.

Decision: retain 011 as a mapper-training checkpoint, not as a deployable model. The main unresolved error is global pose/candidate selection, not residual deformation at the correct plane. Continue the *same* randomly initialized architecture lineage with correct-pose mapper anchoring while fitting evidence trains the pose predictor and candidate selector. Do not claim calibration, animal-level accuracy, or DeepSlice superiority from this readout.
