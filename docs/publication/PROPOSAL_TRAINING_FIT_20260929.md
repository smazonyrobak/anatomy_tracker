# Bounded proposal fit diagnosis

`training/diagnose_joint_v6_training_capture.py` evaluated the completed003
checkpoint in CPU FP32 on 64 evenly spaced original training rows and the first
eight generated training observations. These are **training cases**, not
validation or independent anatomy. The original run used encoder AMP and an
FP32 density head; this diagnostic does not replace its frozen predictions.

| Training subset | Normal-marginal NLL | Normal MAP error | MAP within 10 degrees | Cell top-128 |
| --- | ---: | ---: | ---: | ---: |
| Original fixed64, all | 4.6286 | 32.85 degrees | 20.31% | 50.00% |
| Same, 62 point-supervised | 4.5470 | 31.52 degrees | 20.97% | 51.61% |
| First8 generated, all supervised | 4.5230 | 45.94 degrees | 12.50% | 50.00% |

The eight generated examples are too few to characterize a domain and need not
be easy or uniquely identifiable. Nevertheless, even sampled training images
are not accurately localized: the gap is not exclusively unseen-data accuracy.
Together with the full development audit, this supports a focused normal-loss
control followed by greater image/normal-head expressivity if needed, rather
than making claims from joint NLL improvements alone.

Raw per-row metrics/normal probabilities and checkpoint SHA-256 are retained in
`I:\AnatomyTracker\runs\joint_v6_proposal_curriculum_003_training_capture`.
Fixed, non-selected image panels and learning curves are in the sibling
`joint_v6_proposal_curriculum_003_figures` directory. Truth-selected best-normal
error among the top32 cells is 16.95 degrees on point-supervised synthetic-group
macros; that is an oracle capture diagnostic, not achieved model accuracy.
