# 138 shared anchor-pose corrections: rejected

The 4,000-update, head-only experiment completed from the frozen 132
step-2,000 parent. Each batch contained two independently drawn v4 physical
sections, one v3 section, and acquired coronal/sagittal weak-TRAIN sections.
The 12,000 accepted synthetic physical IDs were distinct. Three heads saw the
same draws: a shared image-conditioned correction, an untied anchor-specific
correction, and a shared correction with its *incremental* image feature
zeroed. The parent, its 16 base modes, all scores, atlas modules and local map
were frozen. The evaluation used the same eight reused synthetic DEV plans
(248 v4 and 243 v3 informative new planes) and six coronal/eight sagittal
weak-DEV donors. No final animals or public benchmark were used.

The predeclared primary metric was the *truth-best* of all 160 rigid proposals
within 0.5 mm, measured across observed valid tissue at native 256 pixels.
This is an oracle of available proposals, not a deployable selector.

| Frozen DEV cohort | 132 parent | Shared visual | Untied visual | Shared source-zero |
| --- | ---: | ---: | ---: | ---: |
| v4, all-160 ≤0.5 mm | 35/248 (14.1%) | **30/248 (12.1%)** | 27/248 (10.9%) | 30/248 (12.1%) |
| v4, all-160 ≤1.5 mm | 200/248 (80.6%) | 200/248 (80.6%) | 200/248 (80.6%) | 199/248 (80.2%) |
| v4, blind-16 ≤0.5 mm | 25/248 (10.1%) | 24/248 (9.7%) | 24/248 (9.7%) | 24/248 (9.7%) |
| v4, all-160 mean | 1.064 mm | 1.078 mm | 1.074 mm | 1.071 mm |
| v3, all-160 ≤0.5 mm | 51/243 (21.0%) | 50/243 (20.6%) | 50/243 (20.6%) | 47/243 (19.3%) |
| v3, all-160 ≤1.5 mm | 212/243 (87.2%) | 211/243 (86.8%) | 209/243 (86.0%) | 212/243 (87.2%) |

The shared visual arm improved the v4 all-160 ≤0.5-mm fraction in **zero of
eight** development plans, far short of the predeclared six-plan and +15-point
requirements. It did not beat either matched control by the required ten
points. Swapping its frozen image feature with that of a different section in
the same plan and appearance class changed v4 capture from 30/248 to 27/248;
this exploratory intervention is not evidence of an anatomy-specific gain.
The training loss difference (terminal shared visual 4.30 versus untied 4.14
and source-zero 4.36 on one draw) therefore cannot justify advancement.

All 491 synthetic first-choice branches were among the **unchanged 16 base
modes**, so their selected errors were unchanged by construction: v4 mean
3.282 mm and ≤1.5 mm 73/248; v3 mean 3.154 mm and ≤1.5 mm 74/243.
The weak-real coronal best-160 donor-equal mean was 0.794 mm for every arm;
sagittal best-160 changed from 0.755 to 0.701 mm for the shared visual arm.
Those references are inherited Allen affines, not expert anatomical truth.
The selected coronal donor regressions relative to frozen 128 persist for
donors 15447 (+0.212 mm) and 15935 (+0.269 mm). Thus the narrow weak-real
best-160 retention condition passes, but the earlier selected-pose guard
still fails.

**Decision:** reject 138; do not promote, extend or tune this anchor-only
head on the same DEV panels. It improved TRAIN objective without improving
independent precise proposals and could not alter first choice. The next
architectural step must change the image-to-pose evidence or the base-mode
selection itself, and demonstrate an anatomy-dependent held-out gain against
support-matched controls before coupled pose/deformation training is scaled.
We should not infer that every arbitrary physical section is uniquely
locatable from one image. No physical steep-oblique expert validation,
calibrated electrode-region probabilities, GUI qualification or DeepSlice
comparison occurred.

Frozen outputs are `I:/AnatomyTracker/runs/shared_anchor_pose_138` and
`I:/AnatomyTracker/runs/shared_anchor_pose_138_dev_eval`. After both processes
exited, a separate audit matched all training and evaluation receipt hashes,
all six step-0/terminal checkpoints, 12,000 distinct accepted synthetic IDs,
1,964 unique synthetic evaluation rows, 888 unique weak-real rows, and the
primary 35→30/248 and 51→50/243 counts recomputed from raw rows. The
evaluator also reproduced the frozen 132 parent rows and 137 all-160 parent
rows before applying the gate.
