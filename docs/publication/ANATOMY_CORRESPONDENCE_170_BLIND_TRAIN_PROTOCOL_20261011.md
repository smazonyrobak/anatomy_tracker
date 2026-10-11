# 170: frozen anatomy correspondence, blind TRAIN candidate selection

Run only after 169 has written `completed.json` and exited. Verify every 169
output hash, its frozen source hashes, the exact 132 step-2000 parent, and
the 64-base synthetic atlas provenance. Evaluate **all** 169 checkpoints
(0, 1,500, 3,000, 4,500, 6,000); do not select a checkpoint using this panel.

Draw 64 new independent TRAIN section attempts from bases 0–59 with a seed
disjoint from 169 training and inner draws. Use the same 80% support-point / 20%
full-box geometry mix and v4 artifact generator. Keep blank attempts in the
record; never retry them. The 121 direct model makes one blind-16 beam per
nonblank section: top eight base-mode branches, top six anchor branches and
two normal-diversity additions. The same branch IDs, image, atlas, slab weights
and candidate starting states are used for every comparison and checkpoint.
Plane truth is read only after candidate scores and choices are fixed.

Compare six fixed selections: prior mass; native 121's better-scored original
or corrected action for each branch; the new
anatomy score before correction; the new score after one bounded correction
and re-render; and two matched atlas-intensity-null ablations at the same
uncorrected/corrected states with support retained. Anatomy scores are used
raw for ranking, **not** tuned or rescaled with the prior on this panel. Report
selected **initial-candidate** 3D rigid-gauge error and capture ≤1.5 mm for
every method. Separately report the selected anatomy-corrected and native-121-
corrected errors; those are not identical geometry and must not be conflated
with candidate selection. Include the blind-16 oracle as a capture ceiling,
and paired gains with section counts by nearest-cardinal angle, observed
tissue-support fraction, and appearance mode. Save every case's branch IDs,
scores, errors, provenance and exact output hashes under I:.

This is a TRAIN-distribution diagnostic, not independent animal validation:
all bases share the same atlas. A genuinely useful feedback signal should
improve blinded candidate selection over prior and native 121 scores, remain
better than the intensity-null ablation, and not collapse on oblique, sparse
or raw-background cases. No probability calibration, GUI promotion, public
benchmark or final animal-level claim follows from this diagnostic alone.
