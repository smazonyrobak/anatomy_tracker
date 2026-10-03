# 043 robust consensus on frozen 041 matches: insufficient

The fixed-seed 128-hypothesis consensus diagnostic completed and its independent read-only verifier passed all row, source, checkpoint, parent, split, identity, donor, and summary checks. It used no learned updates, no truth inside fitting, and no public/final-test images. All 241 development rows and provenance hashes are frozen at `I:/AnatomyTracker/runs/pose_consensus_043_diagnostic`.

Identity-equal synthetic physical error (mm) at 041 batch 1,500:

| Fit | Top one | Oracle best eight |
| --- | ---: | ---: |
| Frozen 019 prior | 2.499 | 1.001 |
| 041 soft, gated | 2.433 | 1.002 |
| 043 hard-match consensus, raw | 2.475 | 0.966 |
| 043 consensus, unchanged 041 gate | 2.457 | **0.950** |

The best-eight gain of 0.051 mm is below the predeclared 0.20-mm direction threshold. On the 112 prior-near-true sections, gated consensus improves 0.681→0.637 mm, again modest. Its consensus inlier fraction is only 26% for the top-ranked candidate and 27% for the physically best candidate, despite 041 soft expectations falling within 1.5 mm of truth for 44%/75% at those ranks. Thus the single best learned key often does not belong to a coherent anatomically correct group. This is more direct evidence that the learned 2D/3D descriptor, not just averaging or one-pass least squares, limits fitting.

The weak-real inherited-affine top-one mean is 0.584 mm prior, 0.618 mm 041 gated and 0.617 mm consensus gated; raw consensus is 0.783 mm. The worst donor regression decreases from 0.108 to 0.056 mm with the unchanged gate, but neither corrected pose beats the weak real reference. These are not expert oblique labels and cannot establish biological accuracy.

Decision: **stop local fitting-rule and threshold tweaks** on 041. The next intervention must improve cross-modal, off-plane correspondence learning and produce a candidate-quality signal that can overturn the overconfident 019 image prior. This follows the distinction established by the scratch-trained 025 patch descriptor: 96.2% synthetic recall@1 on the supplied correct plane, but 026/030–035 showed that simply reusing its score on wrong-plane/global proposals fails. A new model must train with off-plane, hard-lookalike, arbitrary-angle proposals and physical pose/correspondence supervision, not merely add another correct-plane cosine score. [2D3D-MATR](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_2D3D-MATR_2D-3D_Matching_Transformer_for_Detection-Free_Registration_Between_Images_and_ICCV_2023_paper.pdf) and [SLIV-Reg](https://arxiv.org/html/2410.18683) motivate robust matching and orientation handling, but do not validate transfer to histology.

No checkpoint is promoted to GUI use, no numerical uncertainty is calibrated, and no DeepSlice comparison is yet justified.
