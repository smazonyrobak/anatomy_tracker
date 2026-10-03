# 060 fitted-ranker correction: failed development gate

The 6,000-batch run exited normally after 12,000 fresh, independently drawn eligible synthetic TRAIN sections and 6,000 distinct donor-balanced real TRAIN sections. The 059 final pose/encoder/refiner/mapper was frozen; only the 18 fitted-matcher tensors changed. The synthetic target ranked the same eight fitted candidates by observed-tissue 3D mapped error. Real targets ranked them by a five-point disagreement with the weak Allen affine, using a softer distribution because that affine is not expert truth. This run did not use final-test animals, a public benchmark, calibration or GUI promotion.

The frozen evaluator completed all 996 planned rows on the same 185 eligible synthetic DEV sections from eight held-out synthetic identities and 64 weak-real DEV sections from six held-out donors. All values are identity-equal means, not training-batch values.

| Checkpoint | Selected synthetic mapped error (mm) | Exact-pose synthetic mapper error (mm) | Selected weak-real five-point disagreement (mm) | New branch selected, synthetic / real |
| --- | ---: | ---: | ---: | ---: |
| 0 (059 final) | 2.850 | 0.129 | 1.135 | 0.5% / 98.5% |
| 2,000 | 2.958 | 0.129 | 0.687 | 11% / 88% |
| 4,000 | 2.960 | 0.129 | 0.686 | 8% / 86% |
| 6,000 | 2.967 | 0.129 | 0.687 | 11% / 84% |

The real weak-label proxy improved relative to 059, but remained above the 019 reference of about 0.595 mm. All six real DEV donors were still worse than their 059 step-0 (019-weight) readings. Synthetic selection deteriorated, far from the 019 reference of 2.591 mm. The synthetic TRAIN selected-error average also did not improve (2.164 mm in batches 1–1,000 versus 2.249 mm in batches 5,001–6,000), despite a stable approximately 1.16 mm in-beam oracle. This is not merely a held-out overfitting signal; the scorer did not learn useful synthetic selection on its own training stream.

**Decision:** no 060 checkpoint advances and no further scorer-only exposure is justified. Keep 019 as the research reference; do not install 059/060, claim calibrated electrode probabilities, or use the DeepSlice benchmark. The next targeted development should change the global image-to-plane evidence, for example by learning 2D histology-to-3D atlas correspondences or contrastive atlas-plane features from the exact physical synthetic pairs. It should preserve the one-pass pose and local-map output contract and be tested first on a small matched, donor-separated experiment. Any added fit feedback must not be allowed to hide a globally wrong plane behind an elastic warp.

Integrity: run draw, training-log, config, schedule, parent-checkpoint and all six recorded source hashes match; its `completed.json` declares 12,000 accepted synthetic and 6,000 unique real TRAIN presentations. The 203 non-scorer tensors are bitwise identical between steps 0 and 6,000, while all 18 fitted-matcher tensors changed. Evaluator checkpoint, panel, real-record, source, row and summary hashes match. These establish the frozen numerical result and provenance, not biological accuracy.
