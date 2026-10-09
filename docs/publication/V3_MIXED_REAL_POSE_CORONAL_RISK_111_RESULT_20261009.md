# Coronal posterior-risk continuation 111 — audited result

The matched 1,306-batch continuation from 108 step 653 completed. Relative to 110, it changed only one coronal TRAIN loss: student probability mass on branches more than 1 mm from the weak affine incurred proportional excess-error cost. Frozen-teacher retention, shared-feature learning, all stage-two draws, schedules, optimizer state and random states were unchanged. An independent read-only audit matched output/source/checkpoint hashes, byte-identical replayed draws, identity-disjoint TRAIN/DEV records, and raw-row metrics.

| Reused DEV discrepancy (mm) | 108 step 653 | 110 retention only | 111 retention + posterior risk |
| --- | ---: | ---: | ---: |
| Synthetic best of 14, physical | 1.988 | 1.741 | 1.710 |
| Synthetic prior-selected, physical | 3.911 | 3.334 | 3.350 |
| Coronal weak-affine six-donor mean | 0.685 | 1.137 | 0.999 |
| Sagittal weak-affine eight-donor mean | 2.786 | 2.145 | 2.156 |

111 passed the preregistered synthetic gains (0.278 and 0.561 mm) and sagittal gain (0.630 mm; all eight donors improve), but **failed** coronal retention: +0.314 mm exceeds the +0.200-mm limit. It repaired one of 110's two catastrophic coronal branch switches, but another section in the same donor still switched from branch 1 to 0, changing weak-reference error from 0.986 to 11.904 mm. Two of 64 coronal sections now exceed 3 mm, versus none at step 653. The posterior-risk loss helped, but it did not make the selector reliable.

This ends the small matched-loss series on this reused DEV panel. Further tuning coefficients against the same donors risks tailoring the model to them. The next development step should address acquisition-domain and arbitrary-angle data coverage and the image–atlas fit evidence itself, with a fresh identity-disjoint synthetic confirmation before promoting any checkpoint. The selected synthetic error remains about 3.35 mm; the fitter/mapper was frozen in this run, so fitting-to-coordinate learning is still absent. No expert real-animal accuracy, calibrated electrode-region probability, GUI replacement or public benchmark claim follows.

Frozen roots: `I:/AnatomyTracker/runs/v3_mixed_real_pose_coronal_risk_111` and `I:/AnatomyTracker/runs/v3_mixed_real_pose_coronal_risk_111_dev_eval`. TRAIN completion SHA-256 `f16aacc5b6c660aa6bceed641f9835cb9ec3924319f999fb27de2c6dcf25dffe`; evaluation summary SHA-256 `2f701edbf1bb54bcf196c8e07ea347e87d35295af44e4902f37afee38632fa90`; final checkpoint SHA-256 `ad20ca8d642e8aa97370e390ec79b160f9cffa48339e18cfdae1bf00f3859a15`.
