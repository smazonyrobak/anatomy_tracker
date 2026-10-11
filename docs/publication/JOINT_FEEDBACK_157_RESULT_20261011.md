# First matched fitting-to-pose update: negative result

The frozen 155 fitter in the damped 156 model does transmit gradients to the predicted plane: on one on-policy TRAIN case, the corrected-position loss gave nonzero gradients to the predicted state, atlas-derived correspondence and pose encoder; detaching the candidate state left the numerical loss unchanged and removed the pose gradient. `training/probe_joint_feedback_gradient_156.py` records the exact probe.

Two matched arms then saw the same 3,000 new, independently sampled TRAIN sections. Each began from the same 155 checkpoint, used the same direct pose loss and a frozen fitter; only the feedback arm backpropagated the supervised post-fit rigid error into pose. The independent verifier confirmed paired examples, hashes, checkpoint selection and raw arithmetic in `I:/AnatomyTracker/runs/joint_feedback_157_train151_001`.

| Held-out TRAIN synthetic readout, 48 sections / four deformation plans | Initial | Feedback, 2,000 updates | Detached control, 2,000 updates |
| --- | ---: | ---: | ---: |
| Blind first-choice physical error | 3.774 mm | 3.939 mm | 3.728 mm |
| Best of all 160 physical branches | 1.213 mm | 1.322 mm | 1.345 mm |
| Top-16 contains a ≤1.5 mm branch | 29/48 | 25/48 | 25/48 |
| Near-branch error after fitting | 0.939 mm | 0.955 mm | 0.995 mm |

Both arms had their best predeclared selection score at update 0; the feedback arm did not improve blind selection and must not be promoted. Fitting gradients were active for 52.4% of feedback presentations, so this is not a missing-gradient bug. The pose optimizer was restarted without the 154 continuation moments and used one section per update; the matched control also lost candidate coverage. The next targeted comparison will retain the exact 154 pose optimizer moments and accumulate four independent sections per update, keeping the frozen fitter and matched feedback/control design. This is still synthetic, one-atlas evidence and says nothing about calibrated uncertainty or real-animal accuracy.
