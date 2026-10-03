# 044 frozen coarse key-list diagnosis: fine matching is justified

The predeclared 044 diagnostic completed on the unchanged 177 eligible synthetic DEV sections from eight deformation identities. Its independent read-only verifier passed source, protocol, parent/checkpoint, input-panel, row, and summary hashes. No checkpoint was trained or selected, no real weak label or public benchmark was used, and the final-test animals remain untouched. Frozen raw results are in `I:/AnatomyTracker/runs/pose_coarse_topk_044_diagnostic`.

For each valid 16×16 tissue query, an atlas key counts as physically correct when its known synthetic CCF position is within 1.5 mm and atlas support exceeds 0.1. Values below are equal-weighted by synthetic identity; availability means at least one such key exists anywhere in the frozen 041 atlas-key bank. “Best eight” is the physically best of the original eight 019 plane proposals and is an oracle diagnostic, never a deployable selection rule.

| Candidate | Correct key available | Correct in top 8 | Correct in top 32 | Correct in top 64 | ≥3 chart quadrants with top-64 hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 019 image-prior top one | 99.0% | 66.0% | 79.4% | 86.3% | 82.5% of sections |
| Physically best of eight | 100.0% | 95.4% | 99.3% | 99.6% | 87.5% of sections |

The top-K percentages in the table use **all valid query cells** as denominator. Conditional on key availability, top-64 recall is 87.0% for top one and 99.7% for best eight. Both and the best-eight spatial-span fraction exceed the predeclared 60%/80%/80% criteria, respectively. Mean valid tissue queries were only 41.7 of 256 grid cells, so these percentages concern visible tissue and are not a claim of dense whole-frame correspondence.

This changes the engineering target: the coarse bank usually contains a nearby key, but its highest logit is often wrong (043 consensus inliers were only about 26–27%). Training another geometric solver or simply enlarging top-K is unlikely to fix that ranking. The next experiment should train a **candidate-conditioned fine patch matcher on the actual off-plane 019 proposals**, with ground-truth tissue-to-CCF point supervision and off-plane lookalikes as negatives. It must distinguish keys within the short list and learn a separate physical candidate-quality score, because 041's selected branch never left the 019 top-one image prior. This is a development-direction gate, not evidence of biological accuracy or readiness for the GUI. The 1.5-mm threshold is deliberately permissive; subsequent training must report stricter 0.5-mm correspondence accuracy and physical pose errors on new synthetic identities, plus weak-real donor safety, before promotion.
