# Sparse arbitrary-plane capture diagnostic 163 — TRAIN only

This is a frozen diagnostic of the scratch-trained parent-132 pose generator, not an independent-animal test or a qualification result. It asks whether the weak coverage of tiny visible fragments in the 161 feedback-pair preflight is caused mainly by selecting only 16 of 160 predicted branches, or by the absence of a close branch even before selection.

## Exact comparison

- Checkpoint: `I:/AnatomyTracker/runs/v4_pose_adaptation_132/joint_step_02000.pt`, SHA-256 `cc6acacafcbec99f64871cb3a263c9ce3e0b793e5aabd3ae372ed3218c297ee5`.
- Source: frozen 161 TRAIN preflight `I:/AnatomyTracker/runs/joint_feedback_161_train_preflight_001`. Its original draw seeds and base indices were regenerated; the physical-section IDs, valid-pixel counts, checkpoint hash, and each selected beam branch's rigid error were checked against its receipt.
- Cases: all 26 eligible examples with less than 8% valid tissue, from 59 such TRAIN draws. The other 33 were excluded by the predeclared 161 eligibility rule and are not silently treated as successes.
- Metric: mean physical-coordinate error, in millimetres, over the visible valid pixels for each rigid candidate against the known synthetic plane. A branch counts as *near* at at most 1.5 mm, matching the 161 pairing threshold. This is not a biological localization guarantee.
- Code: `training/audit_sparse_plane_capture_163_train.py`. Immutable output: `I:/AnatomyTracker/reports/sparse_capture_163_train_001/rows.json` (SHA-256 `caebd83a540c370098a13e4f14359e71927f5b03185be5d22d0dc7f29f26680f`) and `summary.json` (SHA-256 `0f64b81dcb4a158160f6e0a4a15ce720cf438d843a04b3408fdacda1c8314d01`).

| Candidate set | Near cases / 26 | Median best-branch error |
| --- | ---: | ---: |
| Direct top choice | 2 | not used as a best-of-set summary |
| Blind top 16 by model mass | 6 | 1.962 mm |
| All 160 predicted branches, truth-used only for this diagnostic | 7 | 1.874 mm |

Eleven direct top choices were over 5 mm wrong. Even the truth-selected best of all 160 branches exceeded 3 mm in four cases. Conversely, 19 of 26 eligible sparse examples had *no* branch within 1.5 mm. Enlarging the feedback beam from 16 to 160 would add only one near example in this sample; a better ranker alone cannot repair the other 19.

## Scientific consequence

The on-policy 161 fitter pilot is a legitimate conditional test **when a close candidate exists**. It cannot establish reliable inference for arbitrary brain-intersecting planes or small fragments, because it trains on near/wrong pairs selected from examples that already contain a near candidate. For sparse tissue, the next model-development decision must address candidate coverage and information limits: improve the visual pose proposal and/or allow anatomical/acquisition constraints or multiple sections to disambiguate. If evidence remains insufficient, uncertainty must remain broad or the system must abstain rather than inventing a confident coordinate. These numbers come only from synthetic TRAIN examples generated from one atlas family; no DEV, held-out animal, physical oblique histology, electrode, or GUI claim follows.
