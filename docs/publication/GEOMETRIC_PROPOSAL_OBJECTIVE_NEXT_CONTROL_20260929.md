# Proposed geometric proposal-objective control

Read-only recommendation, 2026-09-29. Not implemented, scheduled, trained or
validated. No active global/local run output trees were accessed. This is a
bounded alternative to another width increase, not an explanation that the
43.73-degree coarse error is acceptable ambiguity: audited median label
quantization is about 3 degrees.

## Preserve probability semantics

`training/run_joint_v6_proposal_curriculum.py:277–282` optimizes joint-cell NLL
plus NLL of its exact normal marginal. Both refer to the same discrete target;
their sum remains a proper composite score, minimized by the true joint
distribution. Single-pose observations do not mathematically force a unimodal
posterior. IPDF demonstrates multimodal orientation learning with one pose per
example. [Murphy et al., ICML 2021](https://proceedings.mlr.press/v139/murphy21a/murphy21a.pdf).

The complete catalogue has 384 antipodal normal classes, 16 support-dependent
offsets per normal, and 16 periodic rolls. Its declared cell masses are uniform,
not uniform density per unit physical offset across differently sized support
intervals. `arbitrary_plane_coarse_proposal_v6.py` already includes these masses
in its normalized probabilities; do not multiply them into predictions again.

## One small geometry-aware control

Keep joint-cell NLL and replace only the auxiliary normal NLL with a kernel
score on the existing 384 classes. For normal probabilities
`p_i = sum_(offset,roll) p(i,offset,roll | image)` and existing target normal
`y = cell_id // 256`, define

```
sigma = 3 * pi / 180
K_ij = exp(-(1 - (n_i dot n_j)**2) / (2 * sigma**2))
S(p, y) = p.T @ K @ p - 2 * (K @ p)[y] + 1
```

Three degrees is a proposed prespecified quantization-scale bandwidth, not a
measured uncertainty or an optimized value. Angles/sigma in the formula are
in radians. With `theta = acos(abs(n_i dot n_j))`, the numerator is
`sin(theta)**2`, not `theta**2`; they agree locally but differ for large angles.
This is a Gaussian kernel on projectors `n n.T`, since their squared Frobenius
distance is `2 * sin(theta)**2`. It respects `n == -n` without identifying
anatomically distinct ML-reflected planes.

For distinct catalogue axes, this kernel is positive-definite. If the true
normal-class distribution is `r`, expected excess score is
`(p-r).T @ K @ (p-r)`, so the target remains `r`, not a broadened distribution.
The **positive self-interaction** `p.T @ K @ p` and **negative cross-term**
`-2 * (K @ p)[y]` are both necessary. Expected geodesic distance alone instead
rewards a point decision and is not a proper full-distribution score.
[Gneiting and Raftery, JASA 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

The fixed matrix is only 384x384. Its softmax-logit gradient has a different
scale from NLL. A multiplier of `384/2 = 192` matches the initial true-logit
gradient for an identity kernel at uniform predictions; it is only approximate
for this nonidentity kernel. If launched, compute the actual initial gradients
from the frozen K and prespecify one aggregate scaling rule before training,
not a scale selected against development accuracy. Retain joint NLL as the
hard-target term, particularly because kernel-score gradients can be small
under confident incorrect predictions. Keep the current pose-censor weights.

## Limits and decision boundary

A narrow kernel changes local error sensitivity; it does not by itself solve
global capture, distant competing modes or appearance mismatch. A very broad
kernel approaches a constant matrix and loses discriminative gradient; it must
not be widened to match the failed model's 43-degree errors. Finite thickness
and small visible sections do not imply one universal angular bandwidth:
information depends on anatomy and observation nuisances.

Ordinary geometric soft-label cross-entropy targets a kernel-smoothed label
distribution instead. It can be studied as regularization, but its prescribed
spread is not calibrated posterior uncertainty. Published label-smoothing
benefits do not establish calibration or accuracy here.
[Muller et al., NeurIPS 2019](https://arxiv.org/abs/1906.02629).

After assessing the existing depth control, change only this objective if a
further control is warranted: same data/schedule, architecture and initialization.
Keep evaluation hard-label NLL, continuous physical angular error, 10-degree
mass/recall and top-eight capture unchanged. The theoretical propriety claim
is for the existing quantized normal labels, not certification of continuous
pose, tissue uncertainty or electrode-site probabilities. Biological animal-level
validation and calibration remain separate requirements.

## Completed catalogue-only scale check

One CPU float64 calculation used the completed 001 `catalogue.pt` only; its
SHA-256 is `9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71`.
It read no images, checkpoints, learned predictions, development metrics or
active output trees. It sampled no random values, used no model RNG, and
selected no parameter from training/development performance. The computation
unit-normalized the 384 stored axes in float64, used sigma = pi/60 radians,
clamped roundoff in `1-dot**2` to zero, and set numerical diagonal entries to 1.

Let `M=384`, `a=K @ ones(M)/M` and `b=mean(a)`. At uniform predicted normal
probabilities, the score's logit gradient for observed class y is

```
g_j_given_y = (2/M) * (a_j - K_jy - b + a_y)
alpha_y = (M-1) / (2 * (1 + b - 2*a_y))
alpha = quantile(alpha_y over all 384 equally weighted classes, 0.5)
```

The selected **single global multiplier is 192.10269359036357**. Per-class
ratios range from 192.03053021002484 to 192.5280443418214; they are inputs to
this geometry-only aggregation, **not class-dependent training weights**.
Class-dependent multipliers generally change the proper-score target. The
full-logit-gradient RMS ratio, 190.5651620346243, was computed for comparison
but was not selected.

The kernel's eigenvalues span 0.501424358575364 to 1.5118994180288383; the
float64 matrix is numerically positive-definite. Analytic/autograd gradient
agreement is within 2.6020852139652106e-18. Median off-diagonal row mass is
0.21605939914747507, so the kernel is local but not exactly identity. Keep it
symmetric, **not row-normalized**. Equivalent score evaluation is
`(p-onehot(y)).T @ K @ (p-onehot(y))`; p remains the normalized normal marginal.
The random initial model is only approximately uniform, and this fixed scalar
matches a specified initialization reference, not every later training state.

Local receipts: [flat calculation](I:/AnatomyTracker/tmp/check_rp2_kernel_scale_20260929.py)
and [numerical receipt](I:/AnatomyTracker/tmp/check_rp2_kernel_scale_20260929.json).
Script SHA-256: `ac7678a01fd162c8c626d3e426e8251c32222d4e587d6ed1182e239600b47baa`.
Kernel float64 byte SHA-256:
`0ab8ae5d734c00ef0d65d540fbaef32b8a3ea7484217683e16cf399d183f4926`.
This check validates the stated finite-kernel arithmetic only; it is no
training, calibration, biological-generalization or registration-quality claim.
