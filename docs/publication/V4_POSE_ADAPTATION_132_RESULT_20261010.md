# 132: dark-section pose capture improved; real-coronal guard failed

The short continuation trained the existing randomly initialized 128 treatment
lineage for 2,000 more batches. Each batch used two newly drawn v4 synthetic
TRAIN sections, one independent v3 section, and acquired coronal and sagittal
TRAIN sections with weak Allen alignment references. Only the shared encoder
and probabilistic pose-head weights were optimized; the atlas matcher and local
mapping weights were frozen. The 6,000 accepted synthetic physical-section IDs
were distinct, with no deliberate same-plane recoloring. All four checkpoint,
configuration, draw-log, training-log and source hashes matched the completed
receipt. This is direct-pose adaptation, **not** evidence that anatomical
fitting now teaches pose prediction.

The matched evaluator compared the parent and all checkpoints on the same
frozen panel of 256 new v4 physical sections from the eight **reused synthetic
DEV deformation plans** (248 informative) and the earlier, non-paired v3 DEV
panel (243 informative). The metric is mean rigid-gauge displacement over
observed valid tissue pixels. “Best of 16” chooses with truth and measures
candidate *availability*; “first choice” uses the model's direct probability
ranking. Neither includes successful local fitting.

| Checkpoint | v4 best-of-16 ≤1.5 mm | v4 first choice ≤1.5 mm | v4 first-choice mean | v3 best-of-16 ≤1.5 mm |
| --- | ---: | ---: | ---: | ---: |
| Parent 128 | 103/248 (41.5%) | 36/248 (14.5%) | 4.440 mm | 193/243 (79.4%) |
| 500 batches | 156/248 (62.9%) | 60/248 (24.2%) | 3.647 mm | 196/243 (80.7%) |
| 1,000 batches | 180/248 (72.6%) | 76/248 (30.6%) | 3.382 mm | 197/243 (81.1%) |
| 2,000 batches | 179/248 (72.2%) | 73/248 (29.4%) | 3.282 mm | 197/243 (81.1%) |

The v4 best-of-16 gain is present in all eight reused DEV deformation plans.
For exposure below 0.15, capture rose from 11/83 (13.3%) to 57/83 (68.7%)
at batch 2,000. At exposure 0.6–1.2 it changed from 40/49 (81.6%) to 39/49
(79.6%). This targets the observed dark-image failure, but a first choice
near the true plane in only about 30% remains poor. The non-paired v3 panel
shows no aggregate proposal loss; the small raw/black/brush strata are
descriptive and not physical-animal evidence.

The predeclared development gate **failed at every checkpoint**. The 1,000-
and 2,000-batch checkpoints cleared the two v4 improvements and v3 retention,
but exceeded the rule that *no* acquired DEV donor worsen by more than 0.20 mm
against its weak five-point Allen affine. At 2,000 batches, coronal donors
15447 and 15935 worsened by 0.212 and 0.269 mm respectively; at 1,000,
donors 15439 and 15935 worsened by 0.245 and 0.269 mm. Donor-equal coronal
disagreement was 0.828 mm in the parent and 0.904 mm at 2,000; sagittal
disagreement improved from 1.490 to 1.242 mm across eight DEV donors.
These inherited alignments are not blinded expert truth. The failed guard
cannot be waived, but it also does not erase the large synthetic capture gain.

**Decision:** retain 1,000 and 2,000 as promising *experimental* pose-bank
checkpoints, not promoted GUI models or calibrated probabilities. Do not extend
the same adaptation unchanged. Next protect real-coronal retention while
addressing the larger remaining first-choice and fitting-signal bottleneck:
the 128/130 atlas scorer did not demonstrate an anatomy-dependent advantage
over its intensity-zeroed control. Any new fitting evidence must beat that
control on matched, support-similar near/wrong planes and show that its loss
actually informs the pose head before joint continuation. Local-map accuracy
with the changed shared encoder also remains untested. No final animals,
expert physical oblique truth, public benchmark, uncertainty calibration or
GUI qualification were used.

Frozen panel, training and evaluator outputs are respectively
`I:/AnatomyTracker/data/fresh_v4_pose_dev_panel_132`,
`I:/AnatomyTracker/runs/v4_pose_adaptation_132`, and
`I:/AnatomyTracker/runs/v4_pose_adaptation_132_dev_eval`. Their completed
receipt SHA-256 values are
`8405da586208dba54a697f7e6e82d02c35c4fe746cfe77a4709eb1b54dda8bbf`,
`daa6157a0e7f400bb532fce9da1a44e5d251ba10822c2afca7d339d891898bf5`,
and `d6e627c0bf4abb17cbd57cc6fe3b52a2ca37d3c4c33d507f7596f66aa1680de2`.
The panel's 256 stored NPZ files, archived source, section provenance and
eligible counts passed a separate post-exit audit; evaluator config, summary
and all raw-row hashes passed the primary post-exit audit. Sections from one
synthetic plan are not independent biological animals.
