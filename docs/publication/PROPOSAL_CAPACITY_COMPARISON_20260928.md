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
