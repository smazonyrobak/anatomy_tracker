# Observed-coordinate continuation 015: mechanism gate failed

This continuation tested whether the previous coordinate head was hampered by supervising an undistorted plane at image cells whose tissue had actually been warped. From the frozen 014 batch-12,000 parent, 015 trained the shared encoder, direct multimodal pose head and coordinate head for 12,000 more batches. Each batch used three fresh arbitrary-plane synthetic sections and one distinct, previously unused Allen TRAIN section: 36,000 accepted synthetic and 12,000 real weak-affine presentations. The atlas-conditioned local mapper and fitted ranker weights remained fixed. Synthetic coordinate supervision emphasized the **actual observed tissue-to-CCF** field, with a low-weight rigid target outside tissue. The direct pose loss and real weak-affine coordinate supervision continued. No external/pretrained weights, old pseudolabels, DEV training, public benchmark or final-test animals were used.

The fixed DEV panel was scored at batch zero and every 2,000 batches: 185 eligible arbitrary-plane synthetic sections from eight synthetic deformation plans, and 64 real sections from six disjoint donors. Plan/donor-equal means in millimetres:

| Added batches | Direct mapped synthetic | Coordinate mapped synthetic | Correct-pose mapped synthetic | Direct real weak-affine | Coordinate real weak-affine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.984 | 2.793 | 0.130 | **0.564** | 0.666 |
| 4,000 | 3.004 | 2.786 | 0.130 | 0.593 | 0.665 |
| 8,000 | **2.799** | 2.706 | **0.129** | 0.619 | 0.688 |
| 10,000 | 2.953 | 2.728 | 0.129 | 0.594 | 0.626 |
| 12,000 | 3.007 | **2.693** | 0.129 | 0.587 | **0.618** |

The best coordinate candidate improved only **3.6%** over the 014 parent (2.793→2.693 mm), below the frozen ≥10% gate. The direct branch did not reliably improve. The true-pose mapper remained near 0.13 mm, so the bottleneck is still global pose. At batch 12,000 the predicted coordinate field is 2.738 mm from the observed local tissue target and 2.733 mm from the rigid target on the same tissue cells—essentially no preferential recovery of the imposed local warp. Severe <5%-tissue cases remain at 4.73 mm mapped error. The coordinate candidate's real weak-affine discrepancy is still worse than the direct branch (0.618 versus 0.587 mm); donor 15447 reaches 0.970 mm. Those weak affines are not blinded expert truth.

**Decision:** 015 fails its preregistered mechanism/continuation gate. Do not scale this same field-loss recipe to a million presentations, promote a checkpoint to the GUI, or interpret its scores as calibrated probabilities. The local observed-coordinate target did not address the multi-millimetre global ambiguity. The next targeted probe must use atlas anatomy to *capture and discriminate* pose candidates over the full brain-intersecting plane range, with a physically meaningful support-aware comparison and fixed-panel controls. Earlier plain intensity correlation and scalar fit scoring failed, so neither can be assumed sufficient. This is development evidence only, not a DeepSlice comparison.

Frozen output: `I:/AnatomyTracker/runs/one_shot_observed_coordinate_pose_015`; fixed DEV: `I:/AnatomyTracker/runs/one_shot_observed_coordinate_pose_015_development_eval`. Training completed normally in 2,448 s. An independent read-only audit matched all pinned source, parent, real acquisition, prior schedule, training-log, checkpoint and DEV-output hashes. It confirmed 12,000 new distinct real TRAIN section identities, 36,000 accepted synthetic draws and recomputed all 63 plan/donor-equal summary values from 1,743 raw rows within 1e-6 µm. Neither final-test animals nor the public benchmark were accessed.
