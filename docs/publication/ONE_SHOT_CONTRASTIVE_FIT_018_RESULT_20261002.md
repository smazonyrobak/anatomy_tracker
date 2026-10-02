# Paired-anatomy feature continuation 018 — development result

The same standalone 017 batch-14,000 model was continued for 8,000 batches: 16,000 fresh arbitrary-plane synthetic presentations and 8,000 distinct TRAIN real sections. The added foreground-paired image/atlas InfoNCE used known synthetic tissue-to-CCF correspondence; candidate physical ranking was weighted more heavily than the training-only exact-pose positive. Atlas rendering, fitting and the fitted-loss gradient into global pose remained connected. No external encoder or legacy weights were loaded.

Frozen outputs: `I:/AnatomyTracker/runs/one_shot_contrastive_fit_018` and `I:/AnatomyTracker/runs/one_shot_contrastive_fit_018_development_eval`. The independent verifier passed source/data/checkpoint hashes, 8,000 training rows, 16,000 accepted synthetic examples, 8,000 distinct real TRAIN sections and 1,245 development rows. There were no public benchmark or calibration uses.

Identity-equal development means (mm):

| Checkpoint | Direct mapped synthetic | Fitted selection synthetic | Best-eight rigid oracle | Exact-pose mapped | Real weak-label fitted five-point |
| --- | ---: | ---: | ---: | ---: | ---: |
| 018 batch 0 = 017 batch 14,000 | 2.727 | 2.789 | 1.198 | 0.126 | 0.542 |
| 018 batch 2,000 | 3.031 | 3.118 | 1.344 | 0.134 | 0.592 |
| 018 batch 4,000 | 2.980 | 3.017 | 1.301 | 0.129 | 0.558 |
| 018 batch 6,000 | 2.886 | 2.851 | 1.268 | 0.130 | 0.579 |
| 018 batch 8,000 | 2.890 | **2.734** | 1.216 | 0.130 | 0.565 |

The best selected result gains 0.055 mm over the trained 017 parent and 0.050 mm over the best 017 initial refined result (2.784 mm). It **fails** the predeclared ≥0.25 mm useful-gain target; direct prediction has become worse, not better, than the parent. Real weak-label donor-mean error remains within the 0.2 mm regression limit, but these inherited Allen affines are not blinded expert truth. The substantial error gap to the exact-pose fitter remains.

The matched candidate diagnostic at 018 batch 8,000 (`I:/AnatomyTracker/runs/one_shot_contrastive_fit_018_candidate_rank_diagnostic`) is hash-bound to the checkpoint/panel/rows. On the same 185 cases, the case-mean fitted choice is 2.778 mm and best-of-eight physical choice 1.199 mm. The fitted scorer selects that physical best in 37.3% of cases versus 34.1% for 017, but mean within-case score/error correlation falls from 0.156 to 0.143. Directly ranking these same branches by the newly trained feature cosine gives 2.880 mm (3.019 mm when multiplied by predicted tissue reliability), so the descriptor is not a valid replacement selection score. Contrastive loss decreasing on training batches did not establish useful held-out anatomical discrimination.

The reserved TRAIN real collection currently has 263,754 images from 1,885 donors. Across all prior pinned schedules, 233,159 distinct sections have already appeared, leaving 30,595 not yet presented. A substantive next exposure stage can use those remaining TRAIN sections first, then explicitly logged replay of prior TRAIN sections if needed; it must never include development, calibration or final-test donors. Synthetic sections continue to be independently randomized rather than deliberate artifact variants of a repeated plane.

Decision: do not add another rerender loop or treat the raw cosine as a score. Retain the same architecture for one substantial exposure test (roughly 30,000 batches), with frozen development evaluation afterward. If anatomical ranking remains flat, reassess model input information and the scoring target rather than indefinitely extending this loss. None of these figures establishes qualified real-animal alignment, calibrated uncertainty, region probabilities or superiority over DeepSlice.
