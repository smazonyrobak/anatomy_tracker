# 094: spatial-verifier development result

**Subsequent correction (096):** the truth-scored correct-match fraction's strong candidate ordering below was mostly reproduced, and slightly exceeded, by leaving points at the fitted plane without using learned match displacements. Its high correlation therefore did **not** demonstrate that the correspondence field contained usable anatomical evidence. See [the 096 geometry-null result](GEOMETRY_NULL_096_RESULT_20261005.md). The frozen 094 measurements remain unchanged.

The prespecified synthetic advancement gate **failed**. Keep the 094 weights as an audited development lineage, not a GUI replacement or calibrated model. The result is informative: training a spatial correspondence head and bounded fit improved the *best available* atlas match, but did not teach the model to choose that match reliably.

## Frozen experiment and audit

- Scratch-only ancestry: 089 → 092 → 094. No outside model weights, pseudo-labels, acquired labels, public benchmark, or final-test animals entered training.
- Training completed 20,000 accepted independently randomized eligible arbitrary-plane sections in 21,935 attempts, with 20,000 distinct recorded physical-section IDs. One appearance was drawn per section. Source/protocol hashes, full draw records, training logs and four checkpoints (0/2,000/8,000/20,000) were verified against the completed receipt.
- Independent evaluation completed the exact fixed 086 beam of 14 branches on 246 eligible 061 synthetic DEV sections/eight plans at all four checkpoints, plus the updated natural beam separately. Its 15,831 raw candidate rows and 984 section-step rows, checkpoint hashes, provenance and result hashes verified. Fine-grid match-head readouts used the 182 sections not inspected in the 093 diagnostic; these remain development data, not untouched final animals.
- Frozen training receipt SHA-256: `1364ab3ea4320424661e8bebe6e2ff79865504d98896a3456c6e56178172b701`. Frozen evaluation receipt SHA-256: `24c27b5f93e9fbc08e9321ea77d6d246b4372381363c5d1be7057a8d4e793b28`. Evaluation candidate SHA-256: `f3d5a39d4a86872fabab65b62a11d0ed70f2c677f59dac4b1e9e2e4015cc777c`; summary SHA-256: `6acc4bf6e3e301722922841f3218230d1d92027b98c02184c13760b1d99dabfe`. Step-20,000 checkpoint SHA-256: `81cd7baeb34bbf5b36a84b3e13987f2794f39389828c9cd0865dd809d6e06815`.

All spatial errors below are plan-equal means on the fixed beam, in millimetres. The 089 comparator used identical 1,024 surviving-pixel mapped96 readout.

| Checkpoint | Selected mapped96 | Best of 14 mapped96 | Score vs lower error ρ | Site correct-match AUROC | Predicted-inlier vs lower error ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 089 inherited comparator | 2.592 | 0.961 | — | — | — |
| 094 step 0 | 2.585 | 0.984 | 0.103 | 0.500 | 0.020 |
| 094 step 2,000 | 2.560 | 0.962 | 0.094 | 0.572 | 0.064 |
| 094 step 8,000, best DEV selected-error checkpoint | **2.511** | 0.898 | 0.088 | 0.574 | 0.104 |
| 094 step 20,000 | 2.537 | **0.864** | 0.086 | **0.606** | 0.075 |

At step 20,000, the selected mapped256 readout was also 2.537 mm; increasing output-grid resolution did not explain the placement error. The fixed gate required selected ≤2.192 mm, score/error ρ≥0.30, best14≤1.011 mm, site AUROC≥0.70 and predicted-inlier/error ρ≥0.30. Only the best14 and appearance nonregression conditions passed. The final step's selected error is 0.055 mm below 089, far short of the required meaningful improvement. Fit-only gradients reached old and anchor pose heads (6.09, 2.82 and 8.97 norm for old, anchor-local and anchor-global on the independent audit section), so the failure is not a severed feedback path.

## Mechanism found in raw candidates

On the 182 complementary DEV sections at step 20,000, each fixed beam's *actual* fraction of correct fine-grid anatomical matches was strongly ordered by physical error (mean within-section Spearman ρ=0.802). It averaged 0.605 for the best physical candidate and 0.114 for the worst. The head's predicted inlier fraction barely changed, 0.858 versus 0.842; its within-section ρ was 0.077. Predicted quality was similarly flat, 0.761 versus 0.744. Thus the useful feedback signal exists in the correspondences, but the learned verifier washes it out. The overall score still follows the direct prior rather than tissue-fit evidence (score/error ρ≈0.086).

This is a stronger explanation than more training loss: the balanced per-batch positive/negative BCE used to train the head is not a probability-calibration objective for candidate inlier *fractions*, and the shallow local verifier has limited global spatial context. These are testable hypotheses, not proof of causation. The correct-match AUROC was still rising at 20,000, so capacity/exposure is not ruled out. [SLIV-Reg](https://arxiv.org/html/2410.18683v1) estimates slice pose from many 2-D/3-D matches with a robust consensus stage; [PointDSC](https://arxiv.org/abs/2103.05465) explicitly uses nonlocal spatial consistency for correspondence rejection. Their results motivate a small targeted comparison of a calibrated fraction/whole-field verifier against the 094 local balanced-BCE head; they do not guarantee transfer to histology.

Next, freeze a matched 095 protocol before any further DEV read: hold the existing synthetic generator, branch beam, parent weights and independent evaluator fixed; change only the verifier objective and global context. Train with prevalence-preserving site BCE plus direct candidate correct-fraction supervision and a whole-grid context path, then check whether head AUROC and candidate-order correlation actually rise before extending exposure. Do not run the public DeepSlice benchmark, claim electrode-region probabilities, or ship to the GUI until independent acquired animal-disjoint evidence supports it.
