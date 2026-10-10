# 140: teacher-forced correspondence before blind analytic plane fitting

## Mechanism question

Can the existing 139 2-D/3-D match-to-affine-solve path learn actual anatomical
correspondences when its descriptors first see a correct or nearby physical
plane? In 139, most queried tissue had no supported key within the 1.5-mm
label radius for a blind candidate; fixed-grid dustbin precision was about
0.90, but mean per-branch physical movement was only 0.075 mm. Its learned
score residual was tiny relative to the frozen pose prior and was insensitive
to source/atlas-intensity ablations. A new rank-only head would repeat the
negative 125/130/134 line, so 140 tests correspondence and a forced analytic
full-frame fit, not branch ranking or direct-pose feedback.

## Frozen model and new training

Freeze the 132 step-2,000 one-shot model, including its image encoder,
probabilistic pose heads, atlas matcher and mapper. Instantiate two identical
fresh `AtlasSpatialFit139` heads: full atlas intensity/support and a matched
support-only control with atlas intensity zeroed. Both see the same source
image, synthetic physical plane, perturbation, atlas support, losses and
optimizer schedule. Do not edit or reuse the trained 139 head. The 139 head's
analytic fit is the only corrected-plane path. Set its scalar fit gate to a
fixed `tanh(2.65) ≈ 0.990` (zero gate weights, frozen bias 2.65), rather than
letting it suppress a bad match, and keep its branch-score head frozen at
zero. Step 0 therefore does **not** reproduce the parent corrected plane;
parent identity is checked separately. No fit score or loss updates the
direct pose predictor in this experiment.
The trainer's true and composed near frames are canonically packed. At
blind evaluation, canonically repack each frozen parent state before passing
it to the 139 head and verify the physical plane agrees with the raw state
within 0.1 µm. This avoids 139's additive 12-parameter fitted-state update
being distorted by an unconstrained raw rotation-6D representation. Compute
the untouched parent baseline on the original raw states.

Train for 4,000 optimizer updates, each with two independently drawn v4 and
one independently drawn v3 informative synthetic TRAIN sections: 12,000
accepted, distinct physical-section IDs. Draw each section once, with one
appearance; never replay it as a recolor or other paired presentation. For
each independent section, use two *training-only* candidate frames: the true
synthetic rigid plane and one small full-frame perturbation chosen near
0.9 mm initial valid-tissue rigid error, accepted only in 0.35–1.5 mm. Sample
up to eight batches of 32 jitters, then use a deterministic 0.9-mm local
translation fallback; record the batch count and update. The near draw
normally perturbs all three rotation and translation coordinates and the
two basis scales plus shear. It does not insert a truth branch into any
development or deployment beam. Its reflection and synthetic PSF are the
section's known generation settings in both arms.

On observed-valid pixels, supervise coarse and local fine 3-D key matching
against the synthetic tissue CCF coordinates, with a dustbin when no
supported key is within 1.5 mm. The 16×16 fixed queries always drive the
analytic fit; independently sampled valid auxiliary queries only add
correspondence labels. Non-tissue auxiliary sites and fixed-grid validity
supervise visibility, not anatomical matches. Updates 1–1,000 use coarse
match weight 1.0, fine 0.5 and visibility 0.2. Updates 1,001–4,000 add
the exact/near corrected rigid full-frame valid-tissue error at weight
0.25, keeping the correspondence losses unchanged. The strong coarse
term targets 139's coarse-top-one miss that starved local fine matches.
Because the 139 fit consumes a hard-selected coarse neighbourhood, report
both supported coarse-positive and fine-positive-given-coarse fractions;
improved cross-entropy without adequate fine-key availability is not a
working correspondence-to-fit mechanism.

Use checkpoints 0, 1,000, 2,000, 3,000 and 4,000 under
`I:/AnatomyTracker/runs/atlas_correspondence_curriculum_140/{full_intensity,support_only}`
as `head_step_XXXXX.pt`. Bind the protocol, parent checkpoint/config and
training source hashes once in `config.json`, log attempted/accepted section
provenance, near initial error, losses and match availability, and write a
completed receipt with config/draw/log/checkpoint hashes. The frozen 132
parent already failed a weak-real coronal guard, so 140 remains an
experimental mechanism test even if its own head improves synthetic fit.

The first launch stopped before its first optimizer update because the default
AMP loss scale of 65,536 produced nonfinite gradients. Preserve that partial
step-0 output separately; it contains no trained head or outcome. Restart the
same frozen protocol with AMP initial scale 256 and growth interval 4,001,
still rejecting any nonfinite gradient rather than silently skipping it.

## Frozen development decision

Before opening 140 results, reproduce the 132/139 parent errors and blind
16 branch IDs on the same 248 informative fresh-v4 and 243 informative v3
DEV sections, with no true or jittered pose supplied to blind inference. Evaluate
all 16 original parent frames and the 140 corrected frames, using native
observed-valid-tissue rigid 3-D error. Report the parent-top-one branch,
truth-best *within the blind 16* near candidate, best corrected candidate,
and selected corrected branch separately, by synthetic plan. For the
truth-best near candidate, also report fixed-grid supported keys, coarse
top-one/top-four hits and local fine hits. No score head is trained, so the
selected branch remains the frozen prior top one; 140 cannot by itself
solve the first-choice selection gap. An exact-truth-pose fit may be shown
only as a clearly labelled diagnostic, never substituted for the blind
readout.

The primary mechanism gate is among v4 sections with a ≤1.5-mm original
branch in the blind 16: at least a ten-percentage-point increase in that
branch's ≤0.5-mm capture after the analytic fit, and at least a 0.20-mm
plan-equal decrease in its rigid error. The full-intensity arm must exceed
the support-only arm by at least five points in conditional ≤0.5-mm capture
and 0.10 mm in plan-equal error reduction. Report whole-panel ≤0.5-mm
availability alongside this conditional gate; the 132 baseline was 25/248.
Require no more than five-point loss of v3 blind-16 ≤1.5-mm capture. Passing
these synthetic tests advances only to source-image/feature swap and
within-support atlas-intensity shuffle on eligible v4 cases. A claim that
anatomy drives the gain requires it to diminish materially under both
ablations while atlas support and candidate frames stay fixed. Acquired
coronal/sagittal weak-affine donor guardrails are required before any later
model promotion or pose-head unfreezing, not for this synthetic-only pilot.
A lower TRAIN correspondence loss or exact-truth diagnostic alone cannot pass.

If this gate fails, stop: do not extend the same head, train a ranker from
its aggregate nine metrics, or backpropagate its fit loss into the direct
pose predictor. If it passes, a separately predeclared blind-beam selection
experiment may use the learned correspondences; only after selection gains
beat support-only and source/atlas ablations should any joint pose update
be considered, with acquired weak-real retention from its first update as
learned from 121/122. The virtual-oblique 113/115/116/117 sections are
weak-affine serial reslices and are not used as an expert all-angle truth
source here. No physical steep-oblique animal validation, probability
calibration, public benchmark, or GUI promotion is implied.
