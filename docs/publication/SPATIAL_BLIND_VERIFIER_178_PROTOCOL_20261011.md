# 178: spatial atlas verifier on the actual blind beam

The 132 randomly initialized lineage proposes a 16-plane blind beam that
often contains a useful plane, yet its first choice remains about 3 mm from
truth. Learned scalar fit scores (170, 171, 176), learned local correlation
peaks (172), and the 174 pose updater have not made a useful blind selector.
The 177 full-intensity 96-pixel local fit did move *oracle-selected nearby*
branches toward truth more often than an intensity-permuted control, but
missed its predeclared physical-gain threshold. This experiment tests one
materially different mechanism: a shared spatial image/atlas comparison on
all 16 imperfect model-generated branches, with one bounded rigid correction
and re-render. It does not yet train nonrigid tissue mapping or pose-head
feedback and cannot by itself qualify the final joint model.

Freeze the 132 step-2,000 model and its existing eight base, six anchor, two
normal-diverse beam rule. Use its original state, reflection, and prior score
for every branch. Render atlas intensity and support with finite slice
thickness and that section's PSF. A small shared two-scale convolutional
network compares contrast-normalized observed image with each 96-pixel
render, retaining spatial layout with local correlation, polarity-aware
difference, and optional brush channels. It predicts a bounded 9D full-frame
update and match score; re-render once and score both original and corrected actions.
Keep the original action available. The visual branch may not receive true
pose, candidate error, branch ID, or atlas coordinates. Synthetic truth is
used only in the training losses and post-hoc evaluation. Avoid using a
single scalar local peak or support area as the match signal.

Train on independently drawn full-sphere TRAIN v3/v4 sections, with raw,
black, and imperfect-brush inputs mixed as one random appearance per physical
section. Reserve TRAIN base indices `3, 11, 19, 27, 35, 43, 51, 59` as the
fixed inner selection panel; train on the other 56. Select 128 independently drawn
panel sections, 64 per appearance, across the eight held-out bases. The 132
parent has seen these TRAIN bases, so this panel selects the *new* module only
and is not animal-independent validation. Record every draw, source/base
lineage, physical plane, appearance/artifact seed, eligible status, and the
same 16 blind branches. No deliberate paired versions of one physical plane.

Train two matched arms from the same random initialization, data sequence,
optimizer and budget: complete real atlas intensity versus complete
within-support intensity-permuted atlas, including the correction re-render.
Both retain identical atlas support and brain-framing information. A
score-stage permutation of the full arm may be diagnostic but is not the
complete null. Use physical five-point error for listwise ranking among all
32 actions and a correction loss on initially 0.5–3 mm,
matching-reflection branches. Keep the direct prior and native 121 fitter as
operational comparators. Penalize correction size and loss of initial atlas
support. For the gate, compare the network's original-branch choice with the
original branch chosen by native 121; report native 121's fitted-action error
separately. Train first for at most 5,000 batches, inspecting a
fixed inner-TRAIN panel at 0, 1,000, 2,500, and 5,000 and retaining its best
checkpoint; do not choose by DEV. Continue exposure only if the fixed panel
and independent fresh TRAIN checks improve, rather than targeting an
arbitrary presentation count.

**TRAIN continuation gate**, separately in v3 and v4: selected *original*
branch mean five-point error at least 0.25 mm below **both** the direct and
native-original-branch comparators; original-branch capture within 1.5 mm at least ten
percentage points above both; and full intensity at least 0.10 mm better
than the complete intensity-null arm. The corrected selected action must not
increase mean physical error. Report corrections of near branches, pairwise
support-matched ordering, nulls, sparse-tissue and steep-oblique strata, and
all base-specific outcomes. Do not pool deliberately enriched strata into the
population estimate. If the gate fails, stop this architecture before DEV
and do not connect its scores to the pose head.

If the TRAIN gate passes, freeze selection and evaluate one fresh synthetic
DEV panel and donor-separated weak acquired readouts with atlas-intensity,
source-swap, and geometry controls. A later matched full-versus-detached
feedback comparison must show that re-rendered fit actually improves the
*same* direct pose distribution, followed by physically constrained
nonrigid-map training and independent animal/electrode calibration. Neither
synthetic labels nor inherited Allen affines prove physical arbitrary-angle
accuracy. No final animals, public DeepSlice benchmark, calibrated
probabilities, or GUI replacement may be used or claimed at this stage.
