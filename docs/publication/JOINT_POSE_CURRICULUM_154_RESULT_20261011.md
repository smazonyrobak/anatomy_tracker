# 154: broader arbitrary-plane proposals, but poor first-choice localization

The 154 continuation finished normally and its independent verifier passed. It
started from the 153 model's **own scratch-trained** step-1,300 checkpoint, kept
its matcher and tissue map frozen, and trained only the direct probabilistic
pose module. The run attempted 59,775 independently randomized TRAIN sections
and accepted 32,000 (8,000 batches of four). Each accepted section had one
plane and one independently drawn appearance; no plane was deliberately
recolored or duplicated. The 60 gradient and four checkpoint-selection
deformation plans are disjoint synthetic identities derived from **one Allen
atlas**, not separate biological animals.

The selected step 8,000 improved proposal *availability* on the fixed 64-case
held-out TRAIN panel. A truth-chosen branch within 1.5 mm occurred in 55/64
cases among all 160 pose/reflection possibilities, versus 8/64 at step zero.
The blind top 16 captured 47/64, versus 1/64. But the blind top two captured
only 16/64, and mean first-choice rigid error remained **4.290 mm** (step zero
4.842 mm). The selection score was the predeclared sum of mean best-of-160 and
first-choice errors, not a fitted or calibrated score.

The separately frozen, *reused* synthetic DEV panels support the same narrow
conclusion. They contain 243 eligible v3 and 248 eligible v4 sections across
eight deformation plans not used for 154 gradient or checkpoint selection.
The evaluator compared 153 and 154 on identical sections, using up to 1,024
fixed valid native-256 pixels per section and all 80 modes × two reflections:

| Model / DEV appearance | Best of 160 within 1.5 mm | Best of top 16 | Best of top two | First choice within 1.5 mm | First-choice mean | First-choice plane-normal error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 153 parent / v3 | 23/243 | 6/243 | 1/243 | 1/243 | 4.735 mm | 56.38° |
| 154 selected / v3 | 193/243 | 155/243 | 68/243 | 41/243 | 3.757 mm | 44.25° |
| 153 parent / v4 | 19/248 | 11/248 | 4/248 | 3/248 | 4.962 mm | 56.32° |
| 154 selected / v4 | 181/248 | 135/248 | 46/248 | 21/248 | 4.338 mm | 47.07° |

The counts are section-weighted; the saved means are equally weighted over
the eight synthetic plans. One earlier internal scratch-trained pose lineage,
132, reached 179/248 v4 best-of-16 and 73/248 first-choice within 1.5 mm on
this panel, with a 3.282-mm first-choice mean, after a longer training history
that included weakly labelled acquired images. The current 154 pose is not yet
better than that research baseline, and neither lineage is suitable for the
GUI. This is contextual, not a matched architecture or exposure comparison.

On the 154 v4 DEV cases with at least 20% valid visible tissue, all-160 and
top-16 capture were 90.8% and 71.1%; with less than 8% tissue they were 55.4%
and 35.7%. The corresponding first-choice means were 3.154 and 5.744 mm.
This supports a real information/coverage problem for tiny fragments, but even
the larger-tissue group is far too inaccurate for electrode-site localization.
For the 32 v4 cases cut at least 45° from the nearest cardinal axis, top-16
capture was 56.2%, versus 52.7% below 30°; these small, synthetic strata do
**not** show an intrinsic steep-angle barrier. V4 top-16 capture across
exposure <0.15, 0.15–0.30, 0.30–0.60 and ≥0.60 was 51.8%, 58.0%, 61.7%
and 46.9%. The earlier extreme dark-versus-bright proposal gap is no longer
monotonic here, but accuracy remains inadequate at every exposure.

The practical bottleneck has shifted from proposal existence toward selection:
on v4, 181/248 sections had a near branch somewhere, yet only 21/248 had one
as the model's first choice. Merely fitting the current top two would miss
most correct candidates. The next stage needs anatomy-dependent fit evidence
over a wider but tractable beam, explicit exact-plane no-harm and deformation
plausibility checks, then a verified gradient path from fitting loss into pose.
Additional direct-pose exposure may be justified while held-out improvement
continues, but more of the same examples or a score that mostly reads atlas
outline cannot close the selection gap. The fit module itself remains weakly
trained and unqualified; 154 did not train it or demonstrate joint feedback.

Neither panel contains an independently cut oblique animal or blinded expert
atlas reference. The existing Allen acquired affines remain weak labels.
There is no trained/calibrated trajectory uncertainty, numerical region
probability, GUI replacement, final animal assessment or DeepSlice comparison.

Frozen training output:
`I:/AnatomyTracker/runs/joint_pose_curriculum_154_train151_001`. Selected
checkpoint SHA-256:
`5a108a464af6c1c769946602ea9144ef0d43760cfe2a67063838f21fbafe86b2`.
Frozen DEV comparison:
`I:/AnatomyTracker/runs/joint_pose_154_frozen_dev_eval_001`; raw rows SHA-256
`0495a4b780a819de925d84916c22cbb4358c0385282f36ff06372b12b60a38f7`.
The read-only verifiers `training/verify_joint_pose_curriculum_154.py` and
`training/verify_joint_pose_154_frozen_dev.py` checked source and output hashes,
all training and evaluation rows, splits, checkpoint selection, and the
reported aggregate arithmetic. Both passed after correcting a parent-hash
comparison typo in the verifier itself; no run artifact was changed.
