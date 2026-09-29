# Proposal depth and normal-readout controls

Source review: corrected normal tilt `1ddce3b`; shared residual encoder
`6ae0bba`. This note records architectural reasoning, not a performance result.
No running 007 outputs, GPU inference, or training code were accessed/changed.

## Why change the learner

The existing 5x5 stem and two stride-two 3x3 convolutions give an 11-pixel
convolutional receptive field at the 24x24 feature map of a 96x96 input.
GroupNorm already communicates global statistics; the deficiency is limited
structured spatial composition, not literally zero global dependence.
An 8x8 pool feeds a 4096-to-64 context projection and eight smooth geometric
mixture components. Perfect eight-case fitting does not establish adequate
anatomical representation across the full arbitrary-plane distribution.
The completed 003 physical-normal error of 43.73 degrees motivates controlled
capacity changes, not more iterations justified only by falling joint NLL.

Residual learning supports trainable depth, but does not establish that our
specific six-block variant will improve registration.
[He et al., ResNet](https://arxiv.org/abs/1512.03385).
SVoRT forms observed/simulated slice pairs, uses ResNet features, and predicts
iterative residual transformations; this supports strengthening image evidence
while retaining re-rendering and correction, not importing its multi-stack
transformer wholesale. [Xu et al., SVoRT, pp.3–4](https://arxiv.org/pdf/2206.10802).
DeepSlice uses Xception, two 256-unit dense layers and 299x299 inputs, but also
ImageNet initialization and primarily coronal training. Its results therefore
do not demonstrate arbitrary-plane performance from random initialization.
[Carey et al., DeepSlice](https://www.nature.com/articles/s41467-023-41645-4.pdf).
Both linked paper PDFs were accessible during this review despite HTML access
failures. No external weights, features or pseudolabels are adopted.

Local implementation evidence is consistent with testing richer features:
`training/atlas_pose_models_v7.py:153–195` uses a deep backbone, spatial pyramid
and 512-dimensional context. Its restricted pose domain and prior training
make it a comparator, not proof or an initializer for the new model.

## Exact controls

The optional 384-normal readout changes only the final normal marginal. For
baseline components `p_l(k)`, weights `w_l`, and normal bias `b[n(k)]`, define
`Z_l = sum_k p_l(k) exp(b[n(k)])`, `p'_l = p_l exp(b)/Z_l`, and
`w'_l = w_l Z_l / sum_j w_j Z_j`. The corrected implementation returns their
mixture, exactly `q(k) = p(k) exp(b[n(k)])/Z`. Thus every within-normal
offset/roll conditional is preserved; all catalogue cells and component
probabilities remain represented. The logsumexp implementation has no detach:
its gradient is also that of this final tilt. At zero bias, baseline gradients
match the original objective up to floating-point arithmetic; after learning,
their change is intentional. Mean-centering the bias removes only a scalar
gauge. This algebra does not diagnose optimization behavior or guarantee gains.

The separate depth option appends six shared 64-channel residual blocks after
the existing encoder: GN(1), GELU, 3x3 convolution, GN(1), GELU, 3x3 convolution,
plus an identity skip. It preserves 24x24 features and atlas-correlation
interfaces while making a 107-pixel convolutional receptive field available.
The final convolution in each block starts at zero, so initial features remain
the old features; the expanded receptive field is learned, not active at step
zero. A forked CPU RNG stream preserves old initialization draws. The same
blocks process histology and rendered atlas rasters inside the single model.
Zero blocks is the unchanged path. There are 444,672 additional parameters.

The root agent's focused two-image check reported exact old parameters/RNG and
initial source features, with finite nonzero gradients on all six final
convolutions (`I:/AnatomyTracker/tmp/check_residual_encoder_20260929.json`).
This is implementation evidence, not evidence of learned registration quality.

Use the bounded head-only control first; the following depth control disables
the optional normal readout to isolate depth. Retain matched data/schedules,
the 64-dimensional context, finite PSF, raw/brush modes and full catalogue.
Only widen context after assessing depth, unless a combined intervention is
explicitly declared. Judge marginal-normal train/development NLL, physical
angle, 10-degree mass/recall and top-eight capture, not joint NLL alone.
Synthetic one-atlas development cannot establish animal-level generalization,
calibration, electrode-site probabilities or superiority to DeepSlice.
