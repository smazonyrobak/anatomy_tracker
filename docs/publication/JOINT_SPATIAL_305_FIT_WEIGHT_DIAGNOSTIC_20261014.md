# 305 blind fit failure is not a scalar-weight tuning problem

A read-only, post-hoc diagnostic reused the verified batch-2,000 `_joint_004` rows for the same 64 synthetic TRAIN-inner sections and 16 truth-blind candidates per section. It varied only the multiplier on the spatial-fit evidence while holding the candidate set, fitted maps, image prior, support-exit penalty and warp penalty fixed. A multiplier of 0 is the recorded support-only score; 1 is the recorded full score. For each of eight synthetic deformation bases, choose the multiplier minimizing mapped-site error on the other seven bases, then score the omitted base. This is exploratory because `_joint_004` checkpoint selection had already used the panel; it cannot qualify a model or establish population accuracy.

| Selection rule | Mean mapped-site error |
| --- | ---: |
| Direct image prior | 2.001 mm |
| Support-only (fit multiplier 0) | 1.917 mm |
| Existing full fit (multiplier 1) | 1.943 mm |
| Leave-one-base-out selected fit multiplier | **2.111 mm** |
| Same held-out multipliers, shuffled atlas intensity | 1.989 mm |

The exploratory training-fold choices among multipliers `{0, 0.125, 0.25, 0.5, 1, 2}` were `{0, 0, 1, 0, 1, 0, 1, 0}` across the eight omitted bases. In-sample mean error was lowest at zero fit weight (1.917 mm); even the base-cross-validated tuning worsened the unchanged direct prior and the shuffled-intensity null. This does not negate the earlier 47/58 support-matched near/far pairwise full-evidence wins: a weak or inconsistent pairwise cue can exist while multi-candidate ranking remains unreliable. It does rule out spending the next development cycle merely searching a global multiplier on this same scalar fit score.

**Decision:** keep the verified fluorescence-plus-old-domain pose lineage, but do not promote the `_joint_004` selector or use its fit score as uncertainty. The next joint experiment must change the *spatial evidence* or its supervision on the candidate bank: require anatomically localized agreement and penalize large/nonphysical warp separately, with a matched support-only and image-intensity-null comparison on fresh identity-disjoint synthetic sections. One-pass plane+map remains the main alternative if the fitted evidence cannot supply an independent gain. Do not scale training, benchmark DeepSlice, or integrate the GUI on this diagnostic.

Read-only source: `I:/AnatomyTracker/runs/joint_spatial_feedback_305_joint_004/readouts.jsonl` (verified by its completion receipt); reproducible computation: `training/diagnose_joint_spatial_305_fit_weight.py`.
