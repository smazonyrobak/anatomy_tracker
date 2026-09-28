# Proposal capacity/resolution comparison

Experiment 002 changes the image encoder from 16 to 64 channels, recurrent
hidden width from 32 to 128, proposal spatial pooling from 4×4 to 8×8, and
proposal input from downsampled 32×32 to native 96×96. This is a combined
capacity/resolution intervention, not an isolated causal ablation. The
recurrent/deformation parameters are still untrained in this proposal phase.

It retains experiment 001's 5,120 training rows, 640 development rows, complete
98,304-cell catalogue, objective, batch size 16, learning rate 0.001, optimizer,
seed and exact row order. The entire new joint model is randomly initialized.
Only prepared images, ground truth, identities and the catalogue are reused;
no model weights are loaded. Their exact file hashes are recorded. Raw complete
development probabilities and full checkpoints are saved every 1,000 updates.

The baseline has failed the internal gate: this comparison is an attempt to
diagnose that failure, not a qualifying model. Compare matching update counts
and elapsed optimization time, not just selected best checkpoints. One seed
does not establish architectural superiority. A wider model need not perform
better, and the observed baseline train/development gap already warns that
data diversity and regularization may also matter.

The primary DeepSlice report used much greater image resolution and a much
larger encoder than our first compact diagnostic
([model-generation methods](https://research-management.mq.edu.au/ws/portalfiles/portal/301063874/Publisher_version_open_access_.pdf)).
This motivates testing capacity; it does not prescribe its pretrained weights,
which are excluded from this project. Our multimodal image-conditioned pose
density is also consistent with the general approach of
[Implicit-PDF](https://proceedings.mlr.press/v139/murphy21a.html), but this
literature does not validate our specific architecture or its accuracy.

All rows derive from one Allen atlas. Existing synthetic-animal IDs are sample
groups, not independent biological anatomies; their macro statistics are
internal diagnostics only. Public benchmarking, claims of calibrated
uncertainty, and release remain deferred.

## Completed outcome

Both runs exited successfully. Their training-row schedules and development
identity files are identical. At step 10,000 the baseline/capacity run have
respectively NLL 9.5803/11.8712, normal error 50.83/49.77 degrees, offset error
2,921/2,819 um, and exact nearest-cell top-128 recall 8.906/4.688 percent.
The baseline applied 10,000 optimizer steps; capacity applied 9,997 of 10,000
attempts (three AMP skips). The capacity loss deteriorated late, so increased
capacity alone is not a successful result. Neither result enables refinement
qualification, calibrated probabilities, benchmarking or release.

`training/compare_joint_v6_proposal_runs.py` writes the matched curves, exact
checkpoint/config hashes and descriptive comparison to
`I:\AnatomyTracker\runs\joint_v6_proposal_comparison_001_002`.
Separate fixed-case diagnosis identified a substantial mixed-precision density
head error. The next controlled continuation starts from capacity step 3,000,
retains its optimizer/scaler/RNG states and remaining row order, and changes
only the head's numerical computation: full precision and removal of
cell-constant embedding offsets before dot products. Its corrected step-3,000
evaluation separates the immediate inference effect from subsequent learning.
Original frozen predictions are retained, not retroactively replaced.
