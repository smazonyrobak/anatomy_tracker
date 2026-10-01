# Stopped v8 all-branch fitting stage — 2026-10-01

The user explicitly stopped training before the planned 30,000-batch stage ended. The `train_joint_v8_allbranch_feedback` process exited after batch 9,296; no matching training process remained. The run is **incomplete**: `completed.json` does not exist. Its 9,296 trace rows represent 18,592 accepted synthetic-plane presentations and 37,184 weak-real-image draws. The run directory and checkpoints at cumulative steps 181,000 and 186,000 remain on I:; nothing was deleted. Do not use this partial run as a final model or treat its terminal exit as a successful completion.

The first versus last 1,000 **training** batches give the following descriptive means. These batches contain only two synthetic sections each and are not a held-out accuracy assessment.

| Training quantity | First 1,000 | Last 1,000 |
| --- | ---: | ---: |
| Chosen fitted five-point error | 4.502 mm | 4.339 mm |
| Best-of-16 fitted error (reference oracle) | 0.880 mm | 0.844 mm |
| Fraction choosing the physically best slab | 0.4845 | 0.4870 |
| Real weak-affine pose error | 0.287 mm | 0.287 mm |

The small change in chosen error and essentially unchanged selection fraction do not establish useful fitting-driven location selection. The best candidate remains far closer than the chosen one. Individual training batches are highly variable. The earlier frozen development readout, before this stage, had chosen arbitrary-plane errors of 5.465–6.761 mm; this partial stage has **no held-out readout**, so its generalization cannot be inferred from the table.

Architecture review found 5,215,622 parameters in the randomly initialized v8 joint model versus 29,899,160 in the frozen AtlasPose architecture. The joint model's 192-pixel input, 4,647,592-parameter image encoder/pose head and approximately 568,030 remaining fitting parameters may limit the harder full-plane task, but parameter count alone is not a causal diagnosis. The current stage fits all 16 branches on synthetic slices; real images only retain the direct pose head using weak Allen affine references. Thus real anatomical fitting and calibration remain untrained. The uncalibrated fitting score must not be reported as a probability.

Next work is design and evidence review only until training is reauthorized: distinguish candidate-generation failure from branch-ranking failure on matched development cases; compare the new model with AtlasPose on the same applicable near-coronal images and physical metric, after checking donor overlap with AtlasPose training (otherwise label the comparison diagnostic only); then decide whether stronger shared image features, fitting comparison and/or finer input resolution are required. Arbitrary-plane performance must be assessed separately. No new training run is authorized by this note.
