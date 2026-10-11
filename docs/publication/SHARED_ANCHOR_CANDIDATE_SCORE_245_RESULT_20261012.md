# Candidate-score diagnosis 245: atlas-sensitive but not useful feedback

## Decision

The [frozen 245 protocol](SHARED_ANCHOR_CANDIDATE_SCORE_245_PROTOCOL_20261012.md)
replayed the same 32 synthetic DEV sections against three frozen 244
checkpoints. **Neither matching the training resolution/beam nor supplying
the intact atlas makes fitted scoring improve blind physical localization.**
Do not continue 244's existing feedback loss, scale this checkpoint, promote
it to the GUI, calibrate electrode probabilities, or run DeepSlice.

The score is numerically sensitive to atlas intensities after batch 1,000,
but sensitivity is not useful ranking: a score-only shifted-atlas control
often selects an equally good or better candidate. The trained deformation
field has shrunk to a few micrometres, so the nominal deformation-difficulty
penalty is effectively absent from the decision. These findings identify a
specific training/design failure, not an impossibility of registering a
physical oblique slice.

## Matched candidate readout

All errors are identity-equal means in mm of physical surviving-tissue CCF
error. “Prior” selects by the image-only prior **within the same already
refined candidate beam**; it therefore differs slightly from the unrefined
prior metric in the 244 result. “Best” is a truth-informed in-beam diagnostic,
not a deployable inference output. The score-only null holds candidate states
and maps fixed and substitutes an 80-voxel-AP-shifted atlas *only in the final
scorer*. Each row has 32 unique sections from eight synthetic plans.

| Batch | Geometry/beam | Prior selected | Fitted selected | Score-only null selected | Best in beam | Fitted pairwise concordance (prior; null) |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| 1,000 | 256 px / 8 | 4.171 | 4.171 | 4.171 | 3.145 | 0.577 (0.577; 0.577) |
| 1,000 | 192 px / 3 | 4.177 | 4.177 | 4.177 | 3.599 | 0.682 (0.682; 0.682) |
| 3,000 | 256 px / 8 | 4.098 | 4.172 | 4.175 | 2.896 | 0.554 (0.570; 0.574) |
| 3,000 | 192 px / 3 | 4.117 | 4.197 | 4.197 | 3.410 | 0.674 (0.663; 0.628) |
| 6,000 | 256 px / 8 | 3.846 | 4.037 | 3.883 | 2.797 | 0.594 (0.534; 0.546) |
| 6,000 | 192 px / 3 | 3.875 | 4.063 | 3.909 | 3.324 | 0.688 (0.740; 0.745) |

Concordance is the within-section fraction of candidate pairs whose score
orders their physical errors correctly, requiring ≥0.25 mm error separation;
ties count half. It is not a calibration metric. The final fitted score
selects the best of eight on only 3/32 sections at batch 6,000; the mean
in-beam gap is 1.239 mm. The readout beam contains a mean 5.8–6.2 distinct
pose modes rather than eight, since reflections sometimes share a mode; the
training beam contains 2.3–2.6. Reflection duplication reduces coverage, but
cannot alone explain why even the matched three-candidate setting worsens
selection.

At batch 3,000 the learned matcher's mean within-beam score SD is 2.20
logit units in the readout geometry, and replacing only atlas intensities
changes its scores by 1.78 logits on average. So the matcher is neither
numerically frozen nor fully atlas-blind. Nevertheless its *prior-excluded*
pairwise concordance is only 0.477, below chance on these near/far candidate
pairs; the intact total score's 0.554 is below the prior's 0.570 and the
score-only null's 0.574. At batch 6,000, total-score concordance rises to
0.594 but top-choice error still worsens by 0.191 mm; a slightly better
average ordering does not establish a reliable top candidate. In the
training-matched geometry, total-score concordance is never consistently
better than the prior across batches, and top-choice error worsens at both
3,000 and 6,000.

The three-candidate geometry is a downsample of the *same* 256-pixel
sections, not a second collection of native 192-pixel specimens. Its
concordance uses far fewer eligible candidate pairs (about 85 across the
panel rather than about 770 for the eight-candidate beam), so small
differences should not be overinterpreted. An independent audit matched
all 245 hashes to the frozen source, checkpoints and panel, reproduced all
192 case selections and pairwise scores from the 1,056 candidate rows, and
found no gross pixel-chart or reflection arithmetic error.

The local field's mean RMS across blind candidates is 5.13 µm at batch
3,000 and 4.74 µm at batch 6,000 in readout geometry. Its score-penalty SD
is only about 0.000005 and 0.000003 logits, respectively, versus learned
match SD around 2.2 logits. The model is **not using how much deformation was
needed** to revise the plane in any material way. The pose-refiner's separate
`delta` score is exactly zero in the replay, consistent with that output
not receiving a ranking loss in 244; it did not cause the failed ranking.

## Mechanism and limits

The 244 training loss ranks three blind candidates together with an injected
exact/near-true pair on only one of two synthetic images per batch. Its
softmax target is scaled by 0.75 mm. When blind candidates are several
millimetres from truth, the injected pair can dominate the target, leaving
little gradient to order *blind candidates against each other*. This is a
mechanistic inference from the code and the measured blind errors, **not** a
measured per-training-batch target-mass statistic. The 245 replay rules out
the 192→256-pixel / three→eight-candidate change as a sufficient explanation:
selection also worsens at the training-matched geometry.

A focused next training comparison should teach atlas-conditioned ranking
*within blind candidates*, while retaining separate near-true local-match
supervision, and must compare intact atlas with an intensity-null/support-only
control. Its local-map supervision must demonstrate gain over a no-warp map
on synthetic deformations large enough to measure, without allowing warp to
hide a wrong plane. That is a proposed correction, not a qualified model.
Before scaling, require a fresh animal-separated development gain in both
candidate capture and blind selection, plus a meaningful deformation gain;
preserve untouched CAL/FINAL animals and the public benchmark.

The replay involved no new gradient updates, real acquired pose truth,
physical heavy-oblique sections, CAL/FINAL animals, uncertainty calibration,
or DeepSlice comparison. Synthetic plans still share one atlas. The run at
`I:\AnatomyTracker\runs\shared_anchor_candidate_score_245_001` contains
192 case records and 1,056 per-candidate rows. Its source, checkpoint, panel
and output bindings are in `completed.json`; its row/case/summary SHA-256
values are `f94e70e9a872a7d614e30bb17ddfb6186191d28ded9b4dccc7eeaf47dbaa4ed3`,
`58d82f0f3f594951e79d0c9ba8477f2e7aac3ea43574f44e87813eeef4dcf35d`
and `ab9a4351e18f2fa52a3825705e4ac5a6c00c97aa6d34d6364ab0157224097bf2`.
