# 284: denser coordinate labels do not rescue the current image mapper

The frozen TRAIN-only run `I:/AnatomyTracker/runs/dense_map_supervision_284_001` exited normally. Its runner/protocol were frozen at commit `7a8cdb4`. The independent read-only `training/verify_dense_map_284.py` passed after one verifier-only correction that normalizes tuple/list provenance before equality; no runner, output, checkpoint, metric, or threshold was changed. The verifier authenticated source/input/output hashes, replayed all 73 panel and 3,309 training attempts, checked the 3,000 accepted unique-section schedule and both arms' target digests, and rerendered every fixed checkpoint readout (64 sections × 4 steps × 2 arms). It did **not** replay the optimizer trajectory.

Both arms saw the same 3,000 independently sampled eligible v9 TRAIN physical sections from bases 0–55, with identical scratch-initialized map weights and the frozen 275 pose features. For this *conditional deformation* test, each section's true synthetic plane and reflection replaced the predicted pose; the test cannot establish blind location or pose–warp feedback. The new readout had eight independently drawn sections per TRAIN base 56–63. Neither DEV/CAL/FINAL animals nor any public benchmark was used.

| Fixed step-3,000 new-base readout | 128 sites per image | Up to 2,048 sites per image |
| --- | ---: | ---: |
| Same-grid no-warp error, mm | 0.212073 | 0.212073 |
| Predicted-map error, mm | 0.209816 | 0.209218 |
| Gain, mm | 0.002256 | 0.002855 |
| Nonnegative base means | 8/8 | 8/8 |
| Mean surviving-site shift, mm | 0.017073 | 0.016888 |
| Maximum post-limit derivative | 0.308473 | 0.320065 |

The dense arm actually used a mean 2,012.6 sites (range 267–2,048); its gain advantage was only **0.000599 mm**, versus the prespecified ≥0.010-mm advantage. Its own gain was far below the ≥0.020-mm signal gate. Both derivative limits passed, and both arms had small positive gains on all eight base means at step 3,000. The dense arm's seven low-visible-support cases had mean gain −0.000297 mm, versus +0.003242 mm on the other 57; these are descriptive small subgroups. Fixed dense-arm gains at 750/1,500/3,000 updates were +0.000155/+0.000911/+0.002855 mm, so the curve was still rising, but **denser labels alone did not solve the learning problem within this matched budget**. Neither arm is promoted or qualified for the GUI.

The source audit also identifies a label-gauge issue that should be measured before scaling: v3 adds random low-frequency image/coordinate warps but retains the pre-artifact affine pose. Some resulting affine motion can be assigned to either pose or deformation, whereas the model's local field projects affine components out. The synthetic pixel-to-CCF map remains coherent for valid tissue, but this pose/deformation split is a generator convention, not uniquely recoverable from one image. Full-sphere normals are proposed, but rejection on visible tissue changes their distribution among accepted images; physical arbitrary-oblique validation does not exist yet. The next bounded work is to quantify this gauge and accepted-plane coverage, then test a mapper whose source and atlas correspondence features can learn together. Repeating the same frozen-feature, site-density schedule at large scale is not justified by 284.

This result does not change the requirement for independent animal-level pose/landmark validation, calibrated electrode-site probabilities, or a later fair DeepSlice comparison.
