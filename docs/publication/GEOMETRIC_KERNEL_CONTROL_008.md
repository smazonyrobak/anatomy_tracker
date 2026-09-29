# Proper geometric normal-objective control 008

Prepared before training; not launched while local joint run 001 owns the GPU.
This is a bounded objective experiment, not a replacement for native joint
learning, animal-level validation or calibrated electrode localization.

Use a fresh complete model with the original shallow encoder and smooth
proposal head, seed 2026092805, and the first 4,000 updates of the same 20,000-step
numeric schedule as 005. Batch 16 remains eight generated complete-catalogue
observations and eight frozen observations, with unchanged brush/appearance
sampling and support censoring. AdamW, learning rate 0.001, clipping and AMP
settings remain unchanged. No pretrained weights or prior model features load.

Keep joint-cell NLL. Replace auxiliary normal NLL with
`alpha * (p_normal - one_hot_y)^T K (p_normal - one_hot_y)`, using the existing
384 normal classes and `y = cell_id // 256`. Build K from unit-normalized
catalogue axes in float64, with
`K_ij = exp(-(1-(n_i dot n_j)^2)/(2*sigma^2))`, sigma = 3 degrees in radians,
and exact diagonal one. Do not row-normalize K or identify anatomical ML
reflections. The original normalized model probabilities are unchanged.

The single global alpha is the median true-logit NLL/kernel gradient ratio at
uniform predictions across every target normal. For `a=K.mean(-1)` and
`b=a.mean()`, ratios are `(384-1)/(2*(1+b-2*a))`; median is
`192.10269359036357` on this catalogue. This scale depends on geometry only,
not images, training losses or development outcomes. The kernel/scaling payload
is saved and hashed before learning. Kernel arithmetic runs in FP32 outside
encoder autocast; normal-marginal NLL remains recorded for comparison but is
not added to this objective. Joint NLL retains useful gradients when the model
is confidently wrong and the kernel score has weak gradients.

The mathematical basis is a proper kernel score, not a smoothed target or an
assertion that a three-degree kernel represents posterior uncertainty.
[Gneiting and Raftery, §5.3, equation 37](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
Its expected excess loss is `(p-r)^T K (p-r)` on the finite quantized classes.
The independent catalogue-only calculation and limitations are in the
[objective note](GEOMETRIC_PROPOSAL_OBJECTIVE_NEXT_CONTROL_20260929.md).

## Fixed decision

Compare the fixed 4k endpoint to 005 at 4k, not to a shorter or longer run and
not to the best development checkpoint. Preserve the earlier engineering
priority threshold: at least five degrees lower marginal-normal error and
at least 0.25 lower normal-marginal NLL before prioritizing more coarse-only
updates. Report physical normal capture within 10 degrees, probability mass
within 10 degrees, nearest-cell recall, joint NLL, and pose/joint and brush
subsets. A failed gate does not prove all geometric scores unsuitable; it means
this fixed experiment does not earn extension. It must not trigger an endless
bandwidth sweep on the same development data.

Do not merge a successful fresh proposal's encoder into an independently
trained local model. A positive result informs the objective of subsequent
whole-model training or a new fully trained lineage. Native local run 001 and
its frozen point-learning criteria remain separate. No public or final-test
benchmark data are used here. One-atlas organizational groups do not establish
generalization across actual animals, even when their IDs are disjoint.
