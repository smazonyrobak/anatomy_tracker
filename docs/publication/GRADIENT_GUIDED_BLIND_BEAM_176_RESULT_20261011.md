# 176 frozen result: two fit-gradient steps do not solve blind plane selection

The stage-1 run completed 500 updates on 1,000 distinct accepted synthetic
physical sections (500 v3, 500 v4; 1,056 attempts) across all 64 TRAIN
deformation bases. The 132 direct pose model and tissue mapper stayed frozen;
only the new five-feature, randomly initialized action ranker trained. Each of
the unchanged blind 16 proposals had a no-shift action and two bounded
gradient steps on finite-thickness, polarity-invariant atlas correlation.
Neither the proposed beam nor the fit used synthetic truth. This experiment
did **not** connect fit loss back to the direct pose head or jointly train the
nonrigid map.

The frozen evaluator used 248 eligible v4 and 243 eligible v3 synthetic DEV
sections from eight deformation plans, plus 64 coronal sections from six
acquired DEV donors and 158 sagittal sections from eight separate acquired
DEV donors. All training output hashes, evaluator input/source/checkpoint
bindings, and evaluator output hashes matched after both processes exited.
An independent aggregation of the 982 synthetic and 444 acquired raw rows
reproduced the reported section-weighted means and unique section counts.

| Synthetic DEV, plan-equal | Original first choice | Trained full-intensity action | Permuted-intensity action | Truth-best original 16 / best 32 actions | Capture ≤1.5 mm, original / trained |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 | 3.278 mm | **3.282 mm** | 3.301 mm | 1.293 / 1.220 mm | 29.4% / 29.4% |
| v3 | 3.149 mm | **3.175 mm** | 3.178 mm | 1.084 / 1.008 mm | 30.5% / 28.4% |

The truth-best columns are availability oracles, **not** blind model
predictions. The fit steps improved that oracle by only 0.07–0.08 mm on
average, although the matched within-support permuted-intensity control
improved it less (to 1.253 mm v4 and 1.058 mm v3). Atlas anatomy therefore
contributed a small conditional local signal, but the learned selector did
not exploit it. Its blind selected error worsened on five of eight v4 and
six of eight v3 deformation plans. The full-versus-permuted blind advantage
was only 0.018 mm v4 and 0.003 mm v3, far short of a material atlas-specific
gain. At step zero, the no-shift action exactly reproduced the frozen direct
prior on both panels, as required by the matched comparison.

The failure is mainly branch choice, not an absence of available proposals.
At step 500 the selector switched from the prior's branch on only 11/248 v4
and 1/243 v3 sections. It chose a corrected rather than no-shift action in
62.5% v4 and 70.0% v3 cases, but this did not improve aggregate placement.
Raw original atlas-fit score alone selected 3.441 mm v4 and 3.109 mm v3;
its permuted-intensity counterpart selected 3.680 and 3.400 mm. This
post-hoc diagnostic suggests some image information exists, not that this
score is a useful global posterior. TRAIN rank loss moved only 1.248→1.222
from the first to last 100 updates, while selected TRAIN error remained near
the frozen prior; the fixed 64-case inner TRAIN check gained 0.17 mm but
captured the same 14/64 sections within 1.5 mm. It did not transfer to DEV.

Acquired weak-Allen donor-equal five-point disagreement worsened from 0.904
to 0.911 mm for coronal and 1.242 to 1.315 mm for sagittal. These inherited
affines are not independent expert anatomical truth; they provide no transfer
support for this ranker.

**Decision:** the predeclared selected-error, capture, and meaningful
atlas-intensity gates all fail. Do not extend this ranker unchanged, promote
its scores as probabilities, start full joint pose/map training from it, or
run DeepSlice benchmarking. The next focused mechanism question is whether
a better-conditioned, trust-region local fit can move *blind* near candidates
substantially closer to truth without privilege or support collapse. Require
a TRAIN-only physical-error and intensity-null gain before learning another
selector; if that fails, improve global direct proposal precision and spatial
matching rather than adding more iterations blindly. The final two-task
fit-to-pose learning connection remains unimplemented and unproven.

Frozen evidence: `I:/AnatomyTracker/runs/gradient_guided_blind_beam_176_train_001`
and `I:/AnatomyTracker/runs/gradient_guided_blind_beam_176_stage1_dev_eval_001`.
The DEV `summary.json` SHA-256 is
`cf8082199e4415a0e4eb0ed2c7f61290c623e365ee1339fd0e959104446d6e99`.
No final animals, calibrated uncertainty, expert physical-oblique references,
GUI replacement, or public benchmark were used.
