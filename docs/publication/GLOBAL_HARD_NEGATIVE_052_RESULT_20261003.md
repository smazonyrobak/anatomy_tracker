# 052 result: local hard negatives did not solve unknown-plane capture

The frozen 052 run continued the scratch-trained 025 two-stem patch descriptor for 5,000 batches of newly sampled TRAIN sections. Each accepted batch contained one independent arbitrary-plane section and 16 local correspondences, contrasted against globally retrieved wrong atlas patches at least 1 mm from the true coordinate. The atlas-only bank held 2,061,824 oriented patches at 4,027 atlas positions. It was rebuilt for the 1,000, 3,000, and 5,000-batch checkpoints without using DEV images.

Evaluation used the same 185 eligible synthetic DEV sections from eight held-out synthetic identities as 030, both image parities, and a fixed seeded set of 32 **true-tissue-valid** query points per section. That query selection is a component diagnostic, not a deployable inference method. The figures below are equal-weight means over held-out identities; a retrieved point counts only when one of the selected bank positions is within 0.5 mm of its known CCF coordinate.

| Encoder / training batches | Global top-1 | Global top-16 | True-plane local top-1 |
| --- | ---: | ---: | ---: |
| 025/030 baseline | 1.86% | 15.42% | 96.22% |
| 052 / 1,000 | 1.86% | 14.54% | 93.05% |
| 052 / 3,000 | 2.02% | 13.49% | 95.98% |
| 052 / 5,000 | 1.54% | 12.33% | 96.89% |

All three checkpoints fail the predeclared global component gate (top-1 at least 5%, top-16 at least 30%, true-plane local top-1 at least 80%, and adequate recall in each raw/black/imperfect-brush stratum). The best global top-16 result is below the unchanged baseline. More local patch discrimination against a small set of mined wrong locations has not yielded reliable whole-brain capture. This does not establish that every contrastive objective or global-context method fails; it rules out promoting this checkpoint or simply extending this exact pilot as a solution.

The evaluator exited normally. Independent read-back verified the evaluator's config, row, and summary SHA-256 receipts and all three rebuilt feature-array receipts. There are exactly 555 result rows (185 sections × three checkpoints), matching the frozen eligible panel. The 5,000 accepted TRAIN presentations have 5,000 distinct physical-section IDs across 64 TRAIN synthetic animals, disjoint from the eight DEV animal IDs; no paired black/non-black duplicate was deliberately generated. Evaluation did not touch final-test animals or a public benchmark, and no probability has been calibrated.

The capture bottleneck remains upstream of fine fitting: the true plane allows strong local matching, but globally plausible false atlas locations dominate the query list. Subsequent work should inspect the whole-slice pose prediction and use spatially distributed anatomical context or a joint pose/map objective; another scalar fit score or the same single-patch mining loop has no supporting DEV evidence. The current model is not ready to replace the GUI model.

Frozen output: `I:/AnatomyTracker/runs/global_hard_negative_052_development_eval`. Atlas bank: `I:/AnatomyTracker/runs/global_hard_negative_052_atlas_bank`. Training: `I:/AnatomyTracker/runs/global_hard_negative_052_pilot`. All artifacts and caches are on `I:`.
