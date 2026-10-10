# 138: shared anchor-conditioned residuals for precise whole-plane proposals

The frozen 137 result leaves only 35/248 frozen v4 sections with any of the
132 parent's 160 rigid proposals within 0.5 mm of the observed valid tissue.
This is a proposal-capture experiment, not fitting-to-pose training. The 132
step-2,000 checkpoint is an experimental parent, not a promoted model: its
coronal weak-reference donor guard already failed. All existing 132 weights,
scores, base-mode states, atlas modules and local mapper remain frozen.

Append a zero-initialized nine-parameter local full-frame correction to each
of the 64 normal-anchor states; retain the 16 base states and both reflection
branches. The treatment applies one shared MLP to each anchor's frame, frozen
image-pyramid features and parent state. Its two same-draw controls use (1)
untied anchor-specific MLP weights on the same inputs and (2) the shared MLP
with its current-image features zeroed while retaining the parent states and
anchor geometry. The latter tests incremental image evidence beyond the
image-conditioned parent. All arms have the same initial 160 physical states,
candidate count, target, loss, optimizer schedule and TRAIN draws. No score is
learned or recalibrated. The parent prior determines first choice; blind-16
quotas remain top-eight base, top-six anchor and two normal-diversity anchors.

The local update right-composes rotation, translation, in-plane scale and
shear. To avoid a capacity-limited negative without consulting 132 DEV, use
the already established conservative 072 component bounds: ±0.9 rad,
±4,000 µm, ±0.25 log-scale and ±0.20 shear. The 063 truth-best mean
centre/normal offsets were 0.893 mm/13.25°, but candidate-specific
parent–truth ranges for 132 were not frozen; these bounds are *not* fitted
quantiles and must not be tuned on the 132 development panel. Unlike 072,
this head has no atlas-correlation or fit pathway, is trained solely for
nearest-anchor full-plane capture, and shares its weights across anchors.
All 160
corrected states are generated solely from the observed image and frozen
parent prediction at inference; no truth pose, affine, mask or label enters.

Run one 4,000-update matched TRAIN experiment: two independently sampled v4
physical sections, one v3 section, one acquired coronal weak-TRAIN section and
one acquired sagittal weak-TRAIN section per update, exactly one appearance per
synthetic physical plane. Log rejected as well as accepted synthetic attempts,
all physical IDs and real donor/section identities. Draw each batch once and
feed it to all three heads; sample the same 128 observed-valid sites for all
arms. For each synthetic section, uniformly supervise the four anchors whose
unoriented frames have normals nearest the true plane. Use the known correct
reflection and the mean 3-D rigid-point displacement on those 128 sites plus
five fixed canvas points, weighted 0.75/0.25, with the 132 normal penalty.
There is no normal-only or truth-best-branch loss. For each weak-real image,
supervise its four nearest weak-affine-normal anchors at the five points, at
half the synthetic loss weight. Weak Allen affines are not expert truth.
Only the heads train, with separate matched AdamW optimizers and the same
4,000-step cosine schedule. Save step 0 and terminal 4,000; terminal is the
sole decision checkpoint. No 137 intensity cue or paired recolor enters.

After training exits, evaluate parent and all three terminal arms on the
unchanged 132 v4 panel (248 informative sections/eight reused synthetic DEV
plans), 128 v3 panel (243 informative), and the same six coronal/eight sagittal
weak-DEV donors. Reproduce the frozen 132 parent rows before decision. The
primary metric is the 137 native-256 observed-valid-tissue *mean rigid-plane*
distance: truth-best of all 160 proposals within 0.5 mm, a diagnostic oracle
of availability, never a selector. Its v4 treatment gate requires at least
15 percentage points above the frozen 132 parent (35/248), at least ten points
above **each** matched control, and a within-plan gain in at least six of
eight plans. v3 all-160 capture at both 0.5 and 1.5 mm may drop by at most
five points versus parent. The weak-real *best-of-160* five-point discrepancy
may worsen by no more than 0.20 mm against parent for **each** donor. Report
the original selected-pose donor guard against frozen 128 separately; a
proposal gate cannot waive 132's existing selected coronal failure.

Report section- and plan-equal all-160 and blind-16 availability at 0.5,
1.0 and 1.5 mm, best-candidate mean/P90 observed-tissue and five-point errors,
first-choice errors, conditional 1.5-to-0.5-mm conversion, and per-plan,
exposure, appearance, tissue-support and nearest-cardinal-angle strata. Record
source-feature swaps as a conditional diagnostic, not a substitute for the
matched source-zero arm. Hash the source, parent, protocol, TRAIN inputs,
draws, checkpoints, panel records and raw DEV rows in I:-only receipts.

A synthetic proposal win would justify a fresh deformation-plan confirmation
and a separately validated anatomy-specific fit signal. It would not
establish selected-pose improvement, real physical-oblique accuracy,
calibrated electrode probabilities, GUI readiness or DeepSlice superiority.
Do not use final animals or the public benchmark in 138.
