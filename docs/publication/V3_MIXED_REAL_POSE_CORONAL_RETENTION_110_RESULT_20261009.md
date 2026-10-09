# Coronal-teacher retention continuation 110 — audited result

The 1,306-batch stage-two continuation from the exact 108 step-653 state completed. It replayed the same training draws byte for byte, updated shared image features as in 108, and added the preregistered frozen-teacher pose-distribution/geometry retention loss on acquired coronal TRAIN images. An independent read-only audit verified source, protocol, checkpoint, schedule and raw-result hashes, 2,612 unique accepted synthetic planes, and TRAIN/DEV identity separation. All numbers below use the same **reused** synthetic DEV panel and weak Allen-affine real DEV references, not expert truth.

| Physical synthetic / weak-affine real discrepancy (mm) | 108 step 653 | 109 frozen-shared step 1959 | 108 unfrozen step 1959 | 110 retained-unfrozen step 1959 |
| --- | ---: | ---: | ---: | ---: |
| Synthetic best of 14 | 1.988 | 1.959 | 1.322 | 1.741 |
| Synthetic prior-selected | 3.911 | 3.856 | 2.949 | 3.334 |
| Coronal six-donor mean | 0.685 | 0.696 | 2.753 | 1.137 |
| Sagittal eight-donor mean | 2.786 | 2.228 | 2.090 | 2.145 |

Against its inherited step-653 state, 110 passed the preregistered synthetic gains (0.247 and 0.577 mm) and sagittal gain (0.642 mm, all eight donors improve). It **failed** coronal retention: donor-equal weak discrepancy rose 0.452 mm, above the allowed 0.20 mm. The loss reduced the catastrophic coronal regression in 108 but did not remove it. Two sections in coronal donor 15439 switched from pose branch 1 to 0 and each became about 11 mm worse; they account for about 75% of the section-weighted regression. Post-hoc exclusion would be invalid, and smaller drift remains across all six donors. A targeted next test is to penalize probability mass on physically bad branches against TRAIN weak references, rather than relying only on a potentially diffuse teacher posterior.

Even a gate pass would not make 3.334-mm selected synthetic error suitable for electrode-site assignment. The atlas fitter and deformation mapper remained frozen; no fitting-to-coordinate learning, real expert landmark accuracy, calibrated region probabilities, GUI replacement or DeepSlice benchmark is established.

Frozen roots: `I:/AnatomyTracker/runs/v3_mixed_real_pose_coronal_retention_110` and `I:/AnatomyTracker/runs/v3_mixed_real_pose_coronal_retention_110_dev_eval`. TRAIN completion SHA-256 `7050343063958b86da028c725467c85005f83bdff9f1bed4b074ed871fa321be`; evaluation summary SHA-256 `dad832590f3a768dd2b4031f9e80267f8aa699a74526f70a69a2ef8ab3f0dcb6`; final checkpoint SHA-256 `de8809deddf619664cd46a61b7a4f3da8616d37e06dbe478d2cb5abf7db32cf2`.
