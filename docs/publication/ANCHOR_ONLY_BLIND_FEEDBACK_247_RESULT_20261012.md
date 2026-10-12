# 247: anchor-only blind-feedback pilot failed its development gate

## Decision

The randomly initialized 13.96M-parameter, 64-antipodal-anchor joint
pose/refinement/local-map model completed its prespecified 6,000 batches on I:
(12,000 independently drawn v7 synthetic sections and 6,000 distinct acquired
TRAIN sections). The [frozen protocol](ANCHOR_ONLY_BLIND_FEEDBACK_247_PROTOCOL_20261012.md)
selected batch 6,000 on the matched 32-section synthetic DEV panel, then read
that checkpoint on all 185 eligible synthetic DEV sections and all 64 weak-real
DEV sections. **Seven of nine continuation gates failed.** Do not scale this
objective, promote this checkpoint to the GUI, calibrate electrode probabilities,
or run the public DeepSlice benchmark.

This is a measured failure of this training design at this exposure, not proof
that arbitrary-plane histology registration is impossible. The local-map and
atlas-feedback mechanisms were present in the same model, but neither produced
a demonstrated useful gain on held-out sections.

## Measured result

All distances below are millimetres. Synthetic distances use surviving-tissue
CCF targets, averaged equally by synthetic deformation plan. Acquired affine
distances are **weak-label disagreement**, not anatomical ground-truth error.

| Checkpoint batch | Matched synthetic fitted rigid | Matched intact-minus-null gain | Matched acquired weak-label disagreement |
| ---: | ---: | ---: | ---: |
| 0 | 4.881 | 0 | 13.241 |
| 1,000 | 4.192 | 0 | 6.942 |
| 3,000 | 4.284 | 0.109 | 7.904 |
| **6,000, selected** | **4.087** | **0.028** | **6.252** |

The selected matched rigid error was 0.050 mm **worse**, not the required
0.400 mm better, than 244's 4.037 mm. Its 0.028 mm intact-atlas advantage
missed the 0.250 mm gate and was positive in only three of eight plans rather
than six. On all DEV sections, selected synthetic rigid error was 4.437 mm,
versus 4.412 mm with shifted-intensity atlas; acquired weak-label disagreement
was 6.368 mm. The truth-informed best of eight synthetic candidates reached
3.896 mm on the full panel, leaving a 0.542 mm selection gap; it is not a
deployable output.

The exact-pose local map was 0.160 mm on the matched panel versus 0.160 mm
with **zero** local warp: a 0.2% relative gain, not the required 10%. On the
full panel it was slightly worse than zero warp (0.13947 versus 0.13938 mm,
plan-equal). The selected-pose full-resolution map was 4.467 mm case-mean,
so it did not repair the pose error. The older DEV generator's small
0.14–0.16 mm zero-warp residual also makes this panel a weak test of whether
large, identifiable local distortions can be undone.

The two superficially passed diversity gates are not evidence of localization:
two anchors account for 152/185 fitted selections (80 and 72), and the full
synthetic normal-axis error is 54.7° mean / 57.8° median. The first and last
1,000 training batches both averaged normal cross-entropy 4.16, essentially
the uniform 64-anchor value `ln(64)=4.159`; no useful orientation learning is
visible in that aggregate. For the 21 eligible sections physically cut
45–55° from the nearest principal plane, fitted rigid error was 4.632 mm
and median normal-axis error 61.9°, failing the 3.5 mm / 20° gates. On the
full panel, 49/185 sections had <10% valid tissue; their mean rigid error
was 5.700 mm, but even sections with 30–50% valid tissue averaged 3.504 mm.

## Readout correction and provenance

The first evaluator execution, preserved as
`I:/AnatomyTracker/runs/anchor_only_blind_feedback_247_readout_001`, wrote all
425 case rows but stopped before a completion receipt. It asserted that 21
cases had a **target atlas tangent-plane** normal 45–55° from a principal
axis; only 20 did. The frozen panel metadata instead has 21 cases with a
**physical cut-plane** normal in that range. Deformation changes the local
atlas tangent normal enough for two sections to cross below 45° and one to
cross above it. The readout-only correction records both angles and uses the
physical cut angle for the prespecified 21-case gate. No checkpoint, model,
training output, source panel, threshold, or case was changed. The failed
readout's 425 rows have SHA-256
`64893953806c1b562ea0c0fa58d2b8673050fd884dcb60aafba96c7805e7ca5f`;
its chosen-checkpoint file has SHA-256
`8c6ced9fe5e218ef0c258c3dcb668b7a69167b955cb4d489c4f6c764dabc4612`.
The corrected completed readout is
`I:/AnatomyTracker/runs/anchor_only_blind_feedback_247_readout_002`.
Its generic angle-stratification table still bins the atlas target-state
normal (20 cases at 45–60°); the prespecified heavy-cut gate bins the
recorded physical cut normal (21 cases). These are different quantities.

The training receipt binds all four checkpoint hashes, all 6,000 batch rows,
all draw identities, 64 TRAIN synthetic bases, 6,000 acquired TRAIN sections,
the frozen protocol and source. Their stored SHA-256 values were rechecked
after exit. The corrected readout's `completed.json` binds 425 raw rows,
chosen batch 6,000, the 185 eligible panel files, six acquired DEV donors,
source, atlas, null control and checkpoint hashes. Raw-row SHA-256 is
`f89069e28606438362b7e14aa384d13c9988f50d49ecf74d2011fd936bfb4b19`;
summary SHA-256 is
`da3a89ca11edd57ff5dff404d8a8798b6b70a81322bf23e3f1066bc79d51be12`.
The model was trained from fresh random weights, with no prior weights,
features or pseudolabels. Neither calibration/final animals nor a public
benchmark were used.

## What the next experiment must isolate

The all-angle 185-section panel is a valid independent synthetic geometry
stress test but uses an older `one_shot` artifact generator, **not v7**.
The matched 32 sections selected the checkpoint and are included in the
185-section readout; their threshold estimates are post-selection, not a
fresh final test. The failure is large enough that this caveat cannot turn
it into a pass.
Prepare the already specified independent v7 DEV panel before attributing
the failure to the architecture rather than appearance shift. Synthetic plans
remain variants of one Allen atlas, not independent biological animals.

Do not simply buy more batches of the same combined 247 loss. First make a
bounded, matched image-only normal/pose signal check on v7 synthetic sections
and compare it with a low-resolution atlas-rendered candidate-retrieval
route; include a shuffled-orientation or intensity-null control and evaluate
on independent v7 synthetic plans plus donor-separated acquired weak labels.
Separately, with the true plane supplied, measure local-map improvement over
zero warp on sections with known, sufficiently large but plausible local
deformations and surviving-fragment targets. Only an approach that improves
both pose capture and identifiable deformation should be trained jointly at
larger exposure. Physical steep-oblique specimen truth and animal-level
calibration are still needed before any arbitrary-cut or electrode-region
probability claim.
