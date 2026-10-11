# 162 result: measured fit-gradient feedback did not improve localization

The paired, TRAIN-only experiment specified in [the frozen protocol](ON_POLICY_JOINT_FEEDBACK_162_POSE_GRADIENT_PROTOCOL_20261011.md) **failed its predeclared progression gate**. Do not promote either checkpoint to DEV, a public benchmark, or the GUI. Frozen run: `I:/AnatomyTracker/runs/joint_feedback_162_pose_gradient_001`.

Both arms started from the same 132-lineage step-2000 weights, saw the same 2,000 independently drawn physical TRAIN sections in the same order, and received the same direct-pose and synthetic near-map supervision. The sole contrast was whether measured atlas-fit/plausibility gradients reached the pose head, branch mass, and shared features (`full`) or stopped at detached predictions (`detached`). An isolated GPU preflight confirmed the intended gradient paths before training. The 665 all-blank draws caused no update in either arm; each made 1,335 paired optimizer updates. The 256 fixed inner TRAIN draws included 83 blanks, leaving 173 brain-intersecting sections for pose scoring. No biological holdout, DEV, or public benchmark was read.

| Frozen inner-TRAIN measure | Full feedback | Detached feedback | Required full-arm improvement |
| --- | ---: | ---: | ---: |
| Selected attempt | 2,000 | 1,500 | — |
| Mean direct first-choice CCF rigid error, 173 sections | 4.246 mm | 4.203 mm | ≥0.15 mm and ≥5% |
| Common-near corrected-map error, 127 sections | 0.010 mm *better* | reference | ≥0.10 mm better |
| Blind-16 capture within 1.5 mm, 173 sections | 76.30% | 75.72% | no >5-point drop |
| Nonzero sparse first-choice difference | 0.061 mm *worse* | reference | no >0.25 mm worsening |

The primary location result is 0.043 mm **worse**, not better, with full feedback. Full also remains worse at equal 2,000-attempt exposure (4.246 vs 4.211 mm). A small common-pair ranking difference (75/92 vs 73/92 native-score wins) does not rescue the failed localization and mapping gates. This result establishes that **this measured loss, gradient route, and data distribution** did not teach better pose; it does not establish that joint feedback is impossible.

The failure has two concrete limitations relevant to the next design. The full-box 151 sampler generated 665/2,000 blank sections, although intended use starts from a brain-intersecting section. Of 346 nonblank sparse (<8% tissue) TRAIN sections, only 116 exposed any near branch and only **three** exposed a support-matched near/wrong pair; therefore the feedback term rarely compared plausible alternatives precisely where global localization was hardest. The four inner pseudo-bases are synthetic TRAIN generators, not independent animals. The next experiment should first use a fresh, independently sampled brain-intersecting-plane stream and verify that the fit signal has a useful correction direction on hard predicted states before another long paired training run. Do not change the frozen 162 records or relax its gates post hoc.

An independent read-only audit matched all 17 output hashes, 17 source hashes, protocol and parent-checkpoint hashes in `completed.json`, verified disjoint ordered draw IDs/seeds and equal updates, and recomputed the failure from the frozen raw logs. The preflight is `I:/AnatomyTracker/runs/joint_feedback_162_gradient_preflight_001/gradient_preflight.json`; the run's `summary.json`, `inner_scores.jsonl`, `training_log_*.jsonl`, and `draws.jsonl` are the authoritative numerical record.
