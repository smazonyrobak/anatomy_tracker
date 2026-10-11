# 178 spatial blind-beam verifier: TRAIN gate fails

The frozen 178 run completed 5,000 one-section optimizer batches, with the
same independently drawn full-sphere TRAIN section and frozen 132 blind
16-plane beam in both arms. The full arm compared the section to each
finite-thickness atlas render with a shared spatial convolutional network;
the matched null arm permuted atlas intensity within the same support at both
the original and corrected renders. Neither arm used a true-plane proposal.
The eight fixed inner-TRAIN bases were excluded from the verifier's gradient
updates. They are **not** new biological animals or a final validation set.

The predeclared continuation gate failed. The best fixed-panel checkpoint was
batch 1,000, selected by the equal-v3/v4 mean error of the *original branch*
chosen by the new score. These are the frozen five-point rigid errors at the
four image corners and centre, in millimetres, on 64 independent sections per
appearance:

| Appearance | Direct prior | Native 121 original branch | Full selected original | Complete intensity-null selected original | Full selected corrected-or-original action | Direct / full capture ≤1.5 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 6.079 | 5.927 | 6.224 | 6.122 | 6.273 | 14.1% / 17.2% |
| v4 | 6.840 | 7.233 | 6.505 | 7.172 | 6.884 | 9.4% / 10.9% |

V4's original-branch mean improves by 0.335 mm versus the direct prior and
0.667 mm versus the complete null, but its capture rises by only one of 64
sections; accepting its predicted correction worsens mean error by 0.379 mm.
V3 selection is 0.145 mm worse than the direct prior, 0.297 mm worse than
native 121, and 0.102 mm worse than the intensity null. Its corrected-action
mean is worse again. Thus neither appearance passes the required joint
mean/capture/null/correction gate. Batches 2,500 and 5,000 also fail; their
equal-v3/v4 panel selected-original means are 6.600 and 6.521 mm versus
6.365 mm at batch 1,000. The fresh TRAIN-base checks do not show a stable
cross-appearance gain.

There is no useful conditional correction signal here. At the selected
checkpoint, matching-reflection branches initially 0.5–3 mm from truth
numbered 130 in v3 and 100 in v4. Their full-arm mean errors changed only
1.969→1.974 and 2.147→2.144 mm; the complete null changed 1.969→1.971
and 2.147→2.133 mm. Full-arm support-matched pair ordering was 54.6% v3
and 56.1% v4, essentially the same as null (54.6%, 55.5%). The source
comparison therefore did not reliably turn atlas anatomy into a better
blind choice or pose correction. Training rank loss nevertheless fell from
about 1.39 in the first 100 batches to 1.21 in the last 100 (null 1.39→1.26),
so lower training loss is not evidence of operational improvement.

The corner-based five-point metric is deliberately strict about the entire
image frame, including corners without observed tissue. The 171/176
diagnostics measured valid-tissue sites instead; their approximately 3-mm
numbers must not be compared directly with this panel's 6–7-mm numbers.
The parent 132 synthetic training cost mixed tissue sites and five frame
points. A strictly **posthoc, non-gate** CPU regeneration of the exact 128
batch-1,000 panel sections selected 128 observed-valid tissue pixels per
section, evenly by raster index as in 171. Every regenerated section ID and
valid-pixel count matched the frozen record. Mean tissue-site errors (mm)
were:

| Appearance | Direct | Native 121 original | Full selected original | Complete null selected original | Direct / full capture ≤1.5 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 2.429 | 2.341 | 2.576 | 2.474 | 46.9% / 48.4% |
| v4 | 3.018 | 3.123 | 2.748 | 2.995 | 40.6% / 46.9% |

The lower absolute scale confirms the metric mismatch, but does not reverse
the decision: v4 has a modest signal, while v3 worsens against both direct
and null selection. These numbers were not used to choose the checkpoint or
revise the gate. The read-only reproducer is
`training/audit_spatial_blind_verifier_178_tissue.py`.

The generator independently randomized
each physical plane; there were no deliberately paired black/background
versions. Its 64 TRAIN bases remain deformations of one CCF, not biological
subjects, and its raw backgrounds/brush/artifact texture are imperfect
proxies for acquired histology.

The independent frozen audit checked all 26 output-file hashes in
`completed.json`, the exact source/protocol/parent bindings, 128 panel
sections (eight per base and appearance), 5,000 accepted gradient sections,
5,228 distinct physical-section IDs, and disjoint 8/56 panel/gradient TRAIN
bases. The run exited normally. The completion receipt SHA-256 is
`4ea33db0d98565998e1d7f37fb7aeb5b4cd0b882b3cdff3303246c26dd6be483`;
the selected batch-1,000 checkpoint SHA-256 is
`f076a9e674e738227a1a5eb9303726381029da1e6027faf56288a8c236f9229c`.
Frozen artifacts: `I:/AnatomyTracker/runs/spatial_blind_verifier_178_train_001`.

**Decision:** do not connect this verifier's score or correction to the
pose head, scale it, open DEV on its behalf, calibrate its probabilities,
replace the GUI model, or run DeepSlice. First determine whether the
all-frame pose target and image simulation match the actual tissue-registration
task; then make one targeted architecture/data change on TRAIN evidence.
This result does not show that arbitrary-plane registration is impossible,
only that this particular spatial verifier is not an adequate solution.
