# 122 joint pose–mapping retention pilot (predeclared)

## Question and lineage

Can the 121 joint encoder, pose, matcher, and atlas-conditioned map improve synthetic blind registration without losing the parent 111 real-image pose behavior? Start from the frozen 111 step-1959 checkpoint, not the failed 121 continuation. The same parent is the frozen coronal teacher. The 121 matcher and 083 auxiliary head are freshly initialized; no external weights or real expert labels are used.

## Fixed pilot

Run 2,000 updates of two independent v3 TRAIN synthetic sections per update. Preserve the 121 direct affine-gauge, blind 16-action, exact/near map, 083 correspondence, bounded warp, and fit objectives. At **every update from 1**, add one reserved TRAIN coronal and one reserved TRAIN sagittal image. Use their inherited Allen affines for the 111-style five-point weak pose loss at coefficient 0.5. On the coronal image, add frozen-parent teacher branch KL plus teacher-posterior-weighted five-point geometry at coefficient 2.0, and student-posterior-weighted `relu(weak five-point error mm - 1.0)` at coefficient 2.0. These real terms update shared encoder/lateral and pose heads only; they do not supervise mapping, matcher, or fit. The affines are weak references, not biological ground truth.

Fit feedback is off through update 1,000; it starts at 1,001 and ramps linearly to the existing maximum 0.05 by update 1,500. Retain 121's best-blind corrected ≤1.5-mm eligibility, support-matched wrong-pair contrast, one-time direct-state gradient-ratio cap, and focused first-eligible fit autograd audit. This exposes a pre-fit real-retention checkpoint and a fit-on checkpoint in one lineage. Checkpoints are 0, 500, 1,000, and 2,000. The pilot stops at 2,000; any longer run needs a separately bound continuation and frozen development decision.

## Frozen evaluation and decision

An evaluator separate from the trainer checks completed receipts and donor/section disjointness, then applies the same blind 14-prior-plus-2-diverse selection and original/corrected actions to parent 111 and each 122 checkpoint. It reports plan-equal selected rigid-gauge and warped-site mapping error on the frozen synthetic DEV panel, exact/near given-pose mapping, proposal oracle error and selection gap, action fractions, and donor-equal five-point weak-affine error for six coronal and eight sagittal DEV donors. Fit diagnostics use a frozen support-matched subset; they are not inference labels or an advancement metric by themselves. No DEV metric enters training or checkpoint gradients.

The 500/1,000 early guardrail requires each real family's donor-equal selected weak-affine mean to stay within +0.20 mm of parent 111; failure means no unqualified continuation. At 2,000, require the same real guardrail, ≥0.30-mm improvement in plan-equal blind selected rigid error versus parent and step 0, ≥0.20-mm improvement in blind mapped-site error versus both, and exact/near given-pose mapping no worse than +0.20 mm versus both. These are the 121 development gates. A fit-to-pose gradient audit proves connectivity only. To claim useful fit feedback before a full continuation, compare the fit-on pilot to a matched no-fit control with identical synthetic/real draws and retention, and require improved blind physical errors without real regression; also test whether the held-out fit-gradient direction locally reduces physical pose error within the ≤1.5-mm basin, stratified by atlas coverage. Lower TRAIN fit or a nonzero gradient alone is insufficient.

All outputs remain on I:. No final-test animals, expert biological truth, calibration, public benchmark comparison, or GUI promotion are authorized by this pilot.
