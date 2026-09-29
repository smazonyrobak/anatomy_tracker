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
