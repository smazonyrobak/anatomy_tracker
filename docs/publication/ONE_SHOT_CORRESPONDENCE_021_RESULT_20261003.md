# Candidate-conditioned correspondence 021 — frozen development result

The isolated 021 test trained a new 119,000-parameter local image/atlas match head for 2,000 batches on 4,000 newly sampled arbitrary-plane synthetic TRAIN sections. It left every inherited 019 pose, warp, atlas and scalar-score weight bitwise unchanged. A training-only known-plane branch and actual predicted top-eight planes supplied pixelwise positives and hard negatives; inference used only the predicted planes. No real DEV, calibration or final-test animal entered training. The independent verifier passed the source, parent checkpoint, 4,000 accepted draw identities, exact frozen-parent weights, five checkpoint hashes and all 1,245 fixed development rows. No public benchmark or probability calibration was used.

All figures below use the *same candidate planes* and the same 96-pixel mapped-tissue endpoint. Synthetic results are equal means over eight synthetic deformation identities, not biological-animal validation. Real results are equal means over six independent donor weak Allen affine labels, not expert ground truth.

| 021 batch | Existing fitted score: synthetic mapping | Spatial score: synthetic mapping | Best of eight mapped oracle | Spatial top-one equals physical best | Real weak-label five-point |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, exact 019 control | 2.590 mm | 2.590 mm | 1.139 mm | 44.2% | 0.595 mm |
| 500 | 2.590 mm | 2.664 mm | 1.139 mm | 42.5% | 0.595 mm |
| 1,000 | 2.590 mm | **2.605 mm** | 1.139 mm | 44.6% | 0.591 mm |
| 1,500 | 2.590 mm | 2.657 mm | 1.139 mm | 43.4% | 0.594 mm |
| 2,000 | 2.590 mm | 2.693 mm | 1.139 mm | 41.3% | 0.592 mm |

The prespecified ≥0.25 mm useful-gain gate **failed**. The best trained checkpoint is 0.015 mm *worse* than the unchanged parent; the final checkpoint is 0.104 mm worse. Real weak-label shifts are small but cannot rescue the failed synthetic selection gate. At batch 1,000 the new score changes 25 of 185 decisions: 11 improve and 14 worsen. The bottom tissue-support quartile worsens by 0.375 mm case-mean, while the remaining three quartiles improve by only 0.090 mm. Even an oracle gate that discarded all low-support corrections would fall short of the 0.25 mm target. The mean within-case score correlation with *negative* physical error is 0.136 for the parent and 0.129 at batch 1,000; the new residual alone reaches only 0.088. Thus the spatial field has some signal on better-supported cases but is not a robust selector.

Training first/last 500-batch means decreased for pixel loss (0.496→0.423) and hard-negative loss (0.477→0.378); predicted-candidate ranking loss barely changed (0.706→0.690). These are unpaired fresh training draws, not evidence of generalization. The fixed development results reject the hypothesis that simply adding this 96-pixel local-correlation head and 4,000 supervised sections closes the pose-selection gap. Do not choose its 1,000-batch checkpoint as a new best, extend the same head blindly, claim calibration, deploy it, or run a public DeepSlice benchmark. Retain 019 batch 18,000 as the current internal checkpoint.

The next change should address the information/capture problem itself: check which tissue-containing planes are physically ambiguous at 96 pixels and whether a higher-resolution or broader candidate representation can distinguish them without a low-support failure. An independent exact-pose control also found only a 0.011 mm average gain from the present local warp over no warp (0.035 mm in the strongest synthetic-warp quartile), so eventual deformation and detached-fragment claims require a harder separate stress panel. All of this remains development work, not qualified real-animal accuracy.
