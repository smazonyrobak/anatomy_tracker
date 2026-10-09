# 117: correcting the virtual training chart was not enough

117 repeated 115's first 1,306 batches from the identical frozen 111 checkpoint. The accepted image/draw log is byte-identical to 115; only the even-batch virtual-oblique pose cost changed from five fixed canvas points to the 16×16 fully donor-valid tissue chart from 116. Loss weight (2.0), physical weak-affine pose labels, optimizer, synthetic and acquired cardinal objectives, teacher retention, and checkpoints were unchanged. Training and the paired evaluator exited successfully. This does not introduce a dense observed-warp target or an explicit normal-angle term.

On the same 32 frozen virtual cuts from eight TRAIN-subset donors, donor-equal means at step 1,306 were:

| Pose discrepancy | 115 five-point cost | 117 tissue-point cost | 117 minus 115 |
| --- | ---: | ---: | ---: |
| Selected, tissue chart | 4.137 mm | 4.042 mm | -0.095 mm |
| Best of prior-filtered 14, tissue chart | 2.551 mm | 2.526 mm | -0.025 mm |
| Oracle best of all 160, tissue chart | 2.054 mm | 2.003 mm | -0.050 mm |
| Selected, five canvas points | 7.925 mm | 8.063 mm | +0.138 mm |

The tissue-only objective moved the intended tissue metric slightly, but it did **not** materially improve proposal capture. At step 1,306, only 3/32 selected 117 predictions were within 2 mm on the tissue chart; even the all-160 oracle was within 2 mm on only 15/32. Neither 115 nor 117 selected an anchored branch on any of the 32 cuts, although the tissue-oracle branch was anchored for 12/32 and 14/32 respectively. This separates a weak selected-branch distribution from the already-large learned proposal floor. It does not establish an intrinsic limit on arbitrary-plane imaging or on the model class.

117 is not a promotion candidate. Its tiny virtual-tissue gain does not justify a long continuation without changing proposal learning and anatomically meaningful fit-to-pose feedback. The independent synthetic and cardinal weak-reference guardrails required before promotion were not rerun, because this branch is not being promoted. These 32 images are weak-affine virtual reslices from TRAIN-subset donors, not acquired oblique slides or untouched biological validation. The chart measures rigid pose discrepancy at donor-valid points, not anatomical landmark or final electrode error.

Frozen run: `I:\AnatomyTracker\runs\v3_virtual_tissue_pose_117`; paired evaluation: `I:\AnatomyTracker\runs\v3_virtual_tissue_pose_117_eval` (256 rows, four checkpoints × two models × 32 cuts). Independent SHA-256 checks passed for all 117 checkpoints, run configuration/draws/training log, and the evaluator's configuration, rows, summary and source. Evaluation rows SHA-256 `647ff2852e23ed08b2bc3e0321e3a46572daad98d68c0c9346468d5bd4804197`; summary SHA-256 `d443799a9ffb544a78cd0ce5c854d7471da8b6a1f7e4c45d89d75ea504555dfc`. No calibration, public benchmark, expert real truth, final animals or GUI deployment are claimed.
