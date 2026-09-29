# Shared spatial depth control, 2026-09-29

Readout006 ended at its predeclared4,000 updates with marginal-normal error
51.06 degrees and NLL5.8459, worse than005 at the same stage. It is not extended.
Changing the output alone did not establish useful orientation learning.

Run007 keeps005's original smooth head (`proposal_normal_readout_count=None`)
and adds six shared residual blocks to the existing image/atlas encoder. Each
block is GroupNorm(1)/GELU/3x3conv twice, with an identity skip and zero final
convolution. Features retain64 channels and24x24 spatial resolution. Learned
convolutional context can span107 input pixels versus11 before the blocks;
GroupNorm already has global statistical dependence, so11 is not an absolute
information boundary. The same blocks process histology and fresh atlas renders.
The recurrent correlation/pose/deformation interface and context width64 stay
unchanged. This is one integrated model option, not a separate inference model.

New branch initialization uses an isolated CPU seed stream
`initial_seed XOR 0x5A17C3`, preserving every old parameter draw and the outer RNG.
Zero final convolutions initially preserve source features exactly. A check on
two existing generated training images confirmed those identities, finite
nonzero gradients at every residual branch output, and444,672 extra parameters.
Receipt: `I:/AnatomyTracker/tmp/check_residual_encoder_20260929.json`.
This is an implementation check, not learned accuracy or generalization.

The run uses a fresh complete model, the same seed2026092805, optimizer,
joint+normal objective and first4,000 updates of005's exact20,000-update schedule.
No prior/external learned weights or features load. Compare fixed4k endpoints,
normal capture and all brush modes; a head-only priority gate of at least5
degrees lower marginal error and0.25 lower normal NLL is reused as an
engineering rule for extending coarse-only training. Do not select a best
intermediate development checkpoint as a test result. If depth alone does not
clearly improve capture, proceed to the prepared native local joint learning
stage and examine optimization rather than repeating long coarse-only runs.

Residual depth is motivated by trainability evidence, not assumed success on
this task: [He et al.](https://arxiv.org/abs/1512.03385).
The blocks start from scratch; no ImageNet, DeepSlice, AtlasPose or previous
project parameters are imported. All datasets and outputs remain onI:.
