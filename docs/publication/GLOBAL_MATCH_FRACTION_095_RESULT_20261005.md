# 095: whole-slice match-fraction readout result

**Subsequent correction (096):** the high truth-scored fraction/error association inherited from 094 was mostly explained by the candidate plane's own geometry, not learned match displacements. See [the 096 geometry-null result](GEOMETRY_NULL_096_RESULT_20261005.md). This strengthens the decision not to train another head on the current match field.

The prespecified synthetic development gate **failed**. Keep 095 as an audited diagnostic, not a model for the GUI. A whole-field head trained to estimate the fraction of anatomically correct matches did not reliably distinguish good atlas-plane candidates from bad ones. More of this head-only training is not justified by these results.

## Frozen run and provenance

- The entire 094 pose, correspondence, fit and local mapping model was frozen at its scratch-lineage step-20,000 checkpoint. Only the 095 candidate-quality head was trained. No external pretrained weights, pseudo-labels, real labels, public benchmark or final-test animals were used.
- TRAIN completed 12,000 accepted independently randomized eligible synthetic sections from 13,231 attempts. Its audited draw log records 12,000 distinct physical-section IDs, one appearance per section, and exact source and protocol hashes. Checkpoints were frozen at steps 0, 4,000 and 12,000.
- The independent evaluator scored all three checkpoints on the exact 086 fixed beam of 14 candidates for 246 synthetic DEV sections/eight plans, plus the updated natural beam separately. It recorded 738 section-checkpoint rows and 12,117 candidate rows. The 182 sections not inspected in the earlier 093 diagnostic supply the focused fraction-quality readout. This is still synthetic development data, not animal-held-out validation.
- The training completion receipt SHA-256 is `e3e29a4c922544546f27f8e7e766dc47f20ed15e51546ef3f9dd2a6270860db2`; the evaluation receipt is `e69a85d8a0b745e3cbbcacaf348dd0d5b75b84049693ccfda56833f8ce049a9d`. The evaluated candidate file is `0a713b2eb323ce2f3d7ab35913eca753a121ccb36d4f7cc2b95ca059ae017253`, and the summary is `2406930e365e7f99a71eda466dd7677cb324de44c7fb2bd240d05eabef9930a8`. Checkpoint SHA-256 at 0/4,000/12,000 is respectively `d57e08be6a99d5af3251afabc4e083fd7896a9ecd165223c35129d08b049b86d`, `fc3acc888015eff6bfe7ae044e34d1cb58d1b4b149f773fb571d25db0efd2e35`, and `4e170fde8f798d5bd824cfac1362eabbe00a87f2ee41e1c0352942b1dbbfae37`. Receipt-declared config, draw, training, checkpoint, candidate and summary hashes verified.

All placement errors are plan-equal fixed-beam means in millimetres; lower is better. Correlations and fraction error in the last two columns use the 182 complementary DEV sections.

| Head checkpoint | Chosen-plane mapped error | Best of 14 mapped error | Score vs lower error ρ, all 246 | Predicted fraction vs lower error ρ | Fraction mean absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.553 | 0.864 | 0.086 | -0.022 | 0.196 |
| 4,000, selected by lowest DEV chosen-plane error | **2.543** | 0.864 | 0.101 | 0.120 | 0.175 |
| 12,000 | 2.570 | 0.864 | 0.097 | 0.142 | 0.170 |

The gate required fraction/error ρ≥0.30, chosen-plane error≤2.192 mm and fraction mean absolute error≤0.15, plus no raw/black/brush stratum over 0.15 mm worse than 089. At the selected step 4,000 all three principal requirements failed; the raw-background stratum also failed. The unchanged 094 best-of-14 error (0.864 mm), exact-black and imperfect-brush conditions passed. Selecting the best of three checkpoints on this DEV beam creates selection bias; the 2.543-mm number is not an unbiased test estimate.

## Interpretation and next decision

On the complementary DEV sections the actual correct-match fraction still separates the physically best and worst branches (about 0.605 versus 0.114 in the 094 audit). By step 12,000 the new predicted fractions were only about 0.298 versus 0.250. The last 1,000 TRAIN batches had lower soft-target cross-entropy than the first 1,000 (about 0.607 versus 0.634), but the pairwise ranking loss did not improve. Thus fitting prevalence modestly better did not make plane selection useful. The unchanged good candidate in the beam rules out candidate *absence* as the sole explanation, but does not establish that the present descriptor/match field contains enough recoverable information to identify it without truth. The next narrow experiment should test cross-modal anatomical descriptor discrimination and spatial consistency directly, with fixed candidates and a frozen independent readout, before another large pose/warp run.

[SLIV-Reg](https://arxiv.org/html/2410.18683v1) and [PointDSC](https://arxiv.org/abs/2103.05465) motivate checking robust geometric agreement across correspondences; they do not validate this model or imply that their reported performance transfers to our arbitrary-plane histology. Do not run the DeepSlice benchmark, make calibrated electrode-region probability claims, or replace the GUI models on the basis of this synthetic diagnostic.
