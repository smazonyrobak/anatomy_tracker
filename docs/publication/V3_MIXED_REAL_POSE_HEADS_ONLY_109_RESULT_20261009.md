# Matched heads-only continuation 109 — audited result

The 1,306-batch control from the exact 108 step-653 model and optimizer completed. Its schedules and every stage-two draw byte matched 108 updates 654–1,959; the inherited model/RNG states matched, all 88 shared encoder/lateral tensors remained unchanged, and only the 12 pose-head tensors changed. Source, protocol, checkpoint, data, raw-row and evaluation hashes all passed an independent read-only audit. The comparison therefore changes shared-feature learning, not training examples or exposure.

| Reused development readout (mm) | Shared frozen, step 653 | Shared frozen, 109 step 1959 | Shared trainable, 108 step 1959 |
| --- | ---: | ---: | ---: |
| Synthetic physical best of 14 | 1.988 | 1.959 | 1.322 |
| Synthetic physical prior-selected | 3.911 | 3.856 | 2.949 |
| Coronal weak-affine six-donor mean | 0.685 | 0.696 | 2.753 |
| Sagittal weak-affine eight-donor mean | 2.786 | 2.228 | 2.090 |

The frozen-feature branch preserved coronal weak-reference agreement within 0.011 mm of step 653 and improved all six coronal donors relative to the unfrozen branch. It did not produce a useful synthetic capture gain: best-of-14 and prior-selected errors fell only 0.029 and 0.055 mm. The matched unfrozen branch gained 0.666 and 0.962 mm over step 653 on those measures but worsened coronal discrepancy by 2.068 mm. Within this training and weak-label DEV setup, **shared-feature updating caused the tradeoff**; more batches or repeated sagittal exposure alone do not explain it. This does not establish that the weak affines are physical truth or that all future unfreezing schemes must fail.

Neither terminal checkpoint is a usable model. The next targeted intervention should permit the shared image features to adapt while retaining the step-653 coronal pose distribution/geometry on acquired coronal TRAIN images, then judge both full-angle synthetic capture and separate-donor coronal/sagittal retention together. If such retention fails, donor-specific virtual oblique training and the image–atlas representation need attention before scaling exposure. The fitter remains frozen in 109: no fitting-to-coordinate learning, calibrated probabilities, expert real-animal accuracy, GUI replacement or public benchmark is demonstrated.

Frozen output roots: `I:/AnatomyTracker/runs/v3_mixed_real_pose_heads_only_109` and `I:/AnatomyTracker/runs/v3_mixed_real_pose_heads_only_109_dev_eval`. TRAIN completion SHA-256 `226fd657adf8a5bdb71496ec91f0a0aefe3538c0b0808c0173aac1677bf4e18d`; evaluation summary SHA-256 `5fa1d4c74d9b04e7e74026eb1c53c493f141bbd9c7cf6f1f5e097b3427874e54`; final checkpoint SHA-256 `d96fcf05ed0070ba36709d34e89c8710a9f7995e9e7baa0b141c1c08981db1ad`.
