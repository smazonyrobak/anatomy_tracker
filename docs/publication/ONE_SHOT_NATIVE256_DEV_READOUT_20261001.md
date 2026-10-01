# Native-256 arbitrary-plane synthetic development readout (1 October 2026)

The completed panel is `I:/AnatomyTracker/data/one_shot_native256_synthetic_dev_001`; its frozen evaluation is `I:/AnatomyTracker/runs/one_shot_atlas_conditioned_warp_native256_dev_001`. The generator used all 32 frozen arbitrary planes in each of four held-out synthetic deformation plans, rerendered at native 256-pixel resolution. Each of the 128 physical sections has exactly one independently seeded appearance (raw background, exact black exterior, or imperfect brush), not three paired copies. Current one-shot tear, fold, missing-tissue, bubble, seam and local-warp sampling was applied. Ninety sections have enough visible tissue to score; 38 remain in the raw output as no-information/censored cases. This is one Allen atlas with synthetic deformation identities, **not** 128 independent biological animals.

The panel/evaluation processes exited without stderr. The evaluation receipt reports 128 rows and 90 eligible, and independent SHA-256 checks match its frozen `rows.jsonl` and `summary.json`. Its source/checkpoint hashes, section identities and input provenance are retained in the receipts. The following means first average sections within each of the four synthetic identities, then average identities; all values are micrometres of visible-tissue-to-known-synthetic-CCF placement error.

| Same eligible 90 sections | Correct pose supplied | Current pose selected by parent model |
| --- | ---: | ---: |
| No learned local warp | 155.24 | 4510.85 |
| Image-only fitter, 4k matched batches | 155.23 | 4510.66 |
| Atlas-conditioned fitter, 2k matched batches | 149.86 | 4510.30 |
| Atlas-conditioned fitter, 4k matched batches | **141.36** | **4511.59** |

At the **correct** pose, the 4k atlas-conditioned fitter improves the no-warp value by 13.88 µm identity-equal. All four synthetic identities have positive paired gains (10.92–17.42 µm), and the result appears in raw, black-exterior and imperfect-brush observations. This supports a modest local-fitting benefit on native 256-pixel, stronger-distortion development images. It does not establish a deployable fit: the same fitter gives no consistent gain at the pose/reflection the model actually selects (identity-equal gain **−0.74 µm**; two identities improve and two worsen). The global error remains roughly **4.51 mm** even after fitting.

The pose/reflection choice was frozen from the parent for every fitter arm, which makes the local comparison fair but also means this experiment cannot test fitting feedback into pose. The 90 eligible sections are synthetic observations, not expert-registered real histology. No uncertainty calibration or public benchmark was used. The result is consistent with the separate physical-metric audit: **pose and branch selection are the dominant failure**, and a ~14 µm local gain at known-correct pose cannot rescue a ~4.5 mm global miss. The next training change must connect candidate-conditioned fitting to predicted pose and rank the same joint pose×reflection branches at training and inference; it must prove benefit from the actual current error range, not only near the truth.
