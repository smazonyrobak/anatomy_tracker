# Damped fitter 156: fresh synthetic development gate

The 155 checkpoint was loaded unchanged into the 156 architecture, which applies 80% of its confidence-limited rigid correction and recomputes the dense map and fit score. Both versions were evaluated on the same 243 eligible, independently generated arbitrary-plane sections from eight synthetic DEV deformation plans in `I:/AnatomyTracker/data/coarse_atlas_pose_150_fresh_dev_panel`. Those plans were not used for the 155 fitter's training or checkpoint selection. The frozen raw rows, configuration and receipt are in `I:/AnatomyTracker/runs/joint_fitter_156_fresh_dev_assay_001`; `training/verify_joint_fitter_156_fresh_dev.py` independently passed the file and arithmetic audit.

| Model | Exact-plane drift | Nearby-plane improvement | Nearby dense-map error | Exact fit preferred to wrong |
| --- | ---: | ---: | ---: | ---: |
| 155, full correction | 0.281 mm | 0.264 mm | 0.633 mm | 228/243 |
| 156, 80% correction | 0.225 mm | 0.256 mm | 0.652 mm | 238/243 |

The predeclared engineering gate was exact drift ≤0.25 mm, nearby improvement ≥0.10 mm and correct-vs-wrong fit discrimination ≥70%; 156 passes, while 155 fails the drift threshold. Damping trades a small amount of nearby correction and dense-map accuracy for less damage to already-correct geometry. These are synthetic sections from a single Allen atlas, **not independent biological animals**, calibrated probabilities, or deployment-quality global prediction. The next experiment must verify that fitting loss reaches the pose predictor during joint training and improves truth-blind candidate selection against a matched no-feedback control.
