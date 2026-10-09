# 122 joint pose–mapping retention pilot: frozen DEV result

The 2,000-update same-architecture pilot completed. An independent post-exit
audit matched its source/protocol/data/parent/checkpoint/output hashes, all
4,000 accepted unique synthetic TRAIN sections, 2,000 coronal plus 2,000
sagittal weak TRAIN presentations, and TRAIN–DEV donor/section separation. All
checkpoints are finite; step 0 preserves the parent model exactly. Fitting
reached the direct pose and shared encoder after update 1,000. Connectivity is
not evidence of useful feedback.

| Checkpoint | Synthetic selected rigid / mapped (mm) | Best of 16 rigid (mm) | Weak real coronal / sagittal (mm) |
| --- | ---: | ---: | ---: |
| Parent 111 = step 0 | 3.091 / 3.082 | 1.500 | 0.999 / 2.156 |
| Step 500, before fit | 3.044 / 3.029 | 1.394 | 1.185 / 2.103 |
| Step 1,000, before fit | 2.933 / 2.925 | 1.360 | 1.159 / 2.052 |
| Step 2,000, fit enabled | 2.901 / 2.896 | 1.318 | 0.922 / 2.004 |

Synthetic means give equal weight to eight virtual plans and 64 held-out
sections; these are not biological animals. Real figures are donor-equal means
over six coronal and eight sagittal DEV donors using inherited *weak* Allen
affines, not expert truth. All real-family non-regression gates pass at 2,000,
unlike 121. No individual DEV donor worsens by more than 0.20 mm at 2,000,
although coronal donor 15935 transiently worsened by about 1.05–1.08 mm at
steps 500/1,000 before recovering. The restored real-training schedule and
teacher/risk terms prevented the gross 121 collapse in this pilot; the run
does not isolate which term was responsible or validate anatomy clinically.

The predeclared 2,000-step advancement gate **fails**: selected rigid error
improves only 0.191 mm against a 0.30-mm threshold and mapped error improves
0.186 mm against a 0.20-mm threshold. The raw result remains useful: exact- and
near-plane-given map errors improve from 0.177/0.785 to 0.144/0.750 mm. But
blind selection still loses 1.583 mm against its own best available action:
49/64 sections contain a ≤1.5-mm candidate, only 21/64 choose one. Thus the
main remaining synthetic error is selection, not absence of a plausible plane.

The present fit difficulty is **not a reliable correction signal**. At step
2,000 a physically worse candidate with matched atlas coverage obtains lower
fit loss in 12/15 diagnostic pairs; this includes 10/13 pairs whose better
candidate is already within 1.5 mm. The fitted checkpoint's small gain over
step 1,000 is confounded with another 1,000 updates; there is no matched
no-fit control. Do not claim that fitting improved pose, increase this fit
weight, or promote 122 to the GUI. A targeted next test must use the same
interior comparison pixels and spatially matched plausible wrong planes,
without an oracle tissue mask at inference, and show anatomical evidence beyond
atlas support before feeding it into the direct pose probabilities. The
real-retention recipe and step-2,000 checkpoint remain available as a guarded
development lineage, not a qualified model.

No expert physical-oblique animal validation, calibration, electrode-region
probability, public DeepSlice superiority, or GUI replacement follows.
Frozen runs: `I:/AnatomyTracker/runs/joint_pose_map_122` and
`I:/AnatomyTracker/runs/joint_pose_map_122_dev_eval`. Evaluation summary
SHA-256: `406e640de7c28d868bb0fdcff90df86c63252655d22522fb1db7ab1be66c0e9c`.
