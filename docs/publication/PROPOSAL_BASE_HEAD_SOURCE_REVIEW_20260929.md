# Base proposal head: targeted source review

Read-only review found no new label-index, quotient-symmetry, or disconnected
gradient-path bug. This is not a proof of correctness or an explanation of the
failed capture gate. No active005 outputs, experiments, GPU work, or benchmarks
were accessed. The optional 384-normal residual readout is **disabled and
untrained in 005**; it is not part of the base-head findings below.

## Contracts checked

- `training/arbitrary_plane_catalogue_v3.py:224,233,249,265` creates canonical
  normal-major, then offset, then roll order:
  `cell_id = ((normal_index * 16) + offset_index) * 16 + roll_index`.
  `arbitrary_plane_training_data_v6.py:187,243` returns the position of the
  minimum geometric cost over these same states. Prepared labels use the exact
  float64 catalogue (`run_joint_v6_proposal_experiment.py:82,121`); generated
  labels are the sampled indices themselves. Thus `label // 256` identifies
  the normal for this bound catalogue, not an offset or roll.
- `arbitrary_plane_coarse_proposal_v6.py:121-127` uses `n nᵀ`, `d n`, and
  `[v, u nᵀ]`. These are invariant to the intended horizontal representation
  change `(u,n,d) -> (-u,-n,-d)`, with `v` unchanged. Off-diagonal projector
  entries retain the distinction under general anatomical ML reflection; the
  head does not silently quotient out left/right brain reflection.
- Images, supplied boundaries, and availability enter the histology stem and
  shared encoder (`arbitrary_plane_recurrent_model.py:258,268,317-337`), then
  proposal context/queries (`arbitrary_plane_recurrent_model_v6.py:297-318`).
  Truth is used for labels/loss, not image channels. There is no detach or
  parameter freeze along this path.

## Gradient and representational hypotheses

For component `l`, let `p_l(k)` be its catalogue softmax, `pi_l` its mixture
weight, and `r_l = pi_l p_l(y) / sum_j pi_j p_j(y)` its responsibility for the
target. The joint-cell negative log likelihood has score derivative
`r_l * (p_l(k) - 1[k=y])` and mixture-logit derivative `pi_l - r_l`
(`arbitrary_plane_coarse_proposal_v6.py:212,226-234`). These derivatives reach
both geometry embeddings and source queries/context/encoder. Nearly identical
components could weaken gate learning; independently randomized query weights
break exact initialization symmetry. This is a collapse **risk**, not observed
collapse or a proven training defect.

Each component score lies in the span of three learned 16-dimensional geometry
embeddings: at most 48 basis functions over 98,304 cells. Its direct normal-only
term has 16 features over 384 normals; offset/roll terms can also influence the
normal marginal, so that marginal is not itself a rank-16 distribution. For
the 64-channel experiment, pooled 64×8×8 features are compressed to 64 context
values before the query heads (`arbitrary_plane_coarse_proposal_v6.py:62-89`).
The log-mixture of eight components is richer than one such score span.

These are concrete architectural restrictions, **not established causes** of
poor capture: even quadratic normal scores can make arbitrarily sharp peaks.
The fixed-eight-case full-head fitting diagnostic is required to distinguish
optimization/expressivity problems from generalization. A 384-normal residual
readout is a focused follow-up hypothesis only; this review makes no claim
that it is necessary, trained, or superior.
