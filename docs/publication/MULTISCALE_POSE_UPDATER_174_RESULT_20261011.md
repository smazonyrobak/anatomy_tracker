# 174 frozen result: multiscale pose correction did not use atlas anatomy

The stage-1 run completed 2,000 optimizer updates (4,000 distinct accepted TRAIN
synthetic sections across all 64 procedural deformation bases). Its source is
commit `f4516a03303693be981a74d8c1ce3d0fd71684eb`; the frozen protocol is
`MULTISCALE_POSE_UPDATER_174_PROTOCOL_20261011.md`. The entire parent 132
model remained frozen. Only the new shared-weight, two-scale, finite-thickness
atlas-conditioned full-pose corrector learned. Training did **not** fit a
local tissue map or update the direct pose head from a fit loss.

The independent frozen DEV evaluator completed normally. Its
`completed.json` records 248 eligible v4 and 243 v3 synthetic sections, plus
64 acquired coronal and 158 acquired sagittal sections. The synthetic panels
reuse eight deformation plans per version; the acquired comparisons use weak
inherited Allen coordinates rather than expert section truth. All frozen
output and checkpoint hashes matched their receipts after the runners exited.
The evaluator declares `calibrated=false`, `expert_real_truth_used=false`,
`final_animals_used=false`, and `public_benchmark_used=false`.

| Frozen DEV panel | Sections | Original selected error | Corrected selected error | Atlas-intensity-null selected error | Original / corrected best-of-16 error | Capture within 1.5 mm, original / corrected |
|---|---:|---:|---:|---:|---:|---:|
| v4 | 248 | 3.282 mm | 3.266 mm | 3.245 mm | 1.293 / 1.263 mm | 29.4% / 29.4% |
| v3 | 243 | 3.154 mm | 3.087 mm | 3.086 mm | 1.084 / 1.056 mm | 30.5% / 30.5% |

The table uses the prespecified blind 16-branch action, including the no-shift
option. “Best-of-16” uses synthetic truth **only to evaluate available
branches**, never to select a branch at inference. The v4 selected gain is
0.016 mm, versus the protocol's 0.30 mm decision target; its best-branch gain
is 0.030 mm, versus 0.20 mm. The null keeps atlas support but removes atlas
intensity. It is better than the full-image correction on v4 and essentially
identical on v3. Thus the improvement cannot be attributed to learning the
matching internal anatomy. Intermediate checkpoints at 500, 1,000, and 1,500
updates did not reveal a checkpoint with material anatomy-dependent gain.

On acquired DEV images, donor-equal disagreement with weak references worsened
from 0.904 to 0.953 mm for six coronal donors and from 1.242 to 1.270 mm for
eight sagittal donors. These numbers are not a biological accuracy estimate,
but they give no transfer evidence in favor of the new corrector.

**Decision:** preserve the experiment, but do not promote or extend this
architecture unchanged, and do not claim fitting feedback, calibration, or
GUI readiness. The failure is specific: a learned atlas-conditioned residual
update can obtain a small geometric damping gain while ignoring atlas
intensity. The next training-only experiment must test whether an explicit,
polarity-invariant finite-thickness fit energy supplies a useful *local
direction* from actual blind model proposals, not merely separates the exact
synthetic truth from a distant wrong plane. Its mask must be constructible at
inference; pairwise comparisons need identical support and an intensity-null
control. Only if that signal is reliable should its differentiable fit loss be
connected to the direct pose head and then jointly trained with a constrained
local tissue map.

Frozen raw evidence:

- `I:/AnatomyTracker/runs/multiscale_plane_updater_174_train_001/completed.json`
- `I:/AnatomyTracker/runs/multiscale_plane_updater_174_dev_eval_001/completed.json`
- `I:/AnatomyTracker/runs/multiscale_plane_updater_174_dev_eval_001/summary.json`
- `I:/AnatomyTracker/runs/multiscale_plane_updater_174_dev_eval_001/synthetic_rows.jsonl`
- `I:/AnatomyTracker/runs/multiscale_plane_updater_174_dev_eval_001/real_weak_rows.jsonl`

The DEV summary SHA-256 in the completion receipt is
`0e4edf5480d3076f2c52328ed5d7deff7d6030d64bcdf9ec05766127611f224b`.
