# 128: in-path correspondence learned, but did not fix blind localization

The matched 128 comparison is complete. Both arms continued the same frozen
122 step-2,000 model for 6,000 batches, presenting the same 12,000 independently
drawn synthetic TRAIN sections, 6,000 weak-reference coronal TRAIN images and
6,000 weak-reference sagittal TRAIN images. The treatment alone added a 0.003
cross-entropy loss to the existing global matcher's image–atlas attention.
The control and treatment draw hashes match. The fresh development panel has
256 independently drawn physical synthetic sections on the same eight prior
synthetic DEV deformation plans; 243 were eligible. These are neither new
biological animals nor physical oblique sections.

The treatment learned the added training task: its final 1,000-batch mean
correspondence loss was **4.763**, against **6.240** for control on the same
draws. This did not establish useful anatomical evidence for plane selection.

| Fresh synthetic DEV, final step 6,000 | Frozen 122 | Matched control | Treatment |
| --- | ---: | ---: | ---: |
| Plan-equal selected mapped-tissue error (mm) | 3.327 | 3.322 | 3.244 |
| Plan-equal selected within 1.5 mm | 23.5% | 27.1% | 27.1% |
| Section-weighted selected mapped-tissue error (mm) | 3.329 | 3.320 | 3.243 |
| Section-weighted selected within 1.5 mm | 23.5% | 27.2% | 27.2% |
| Best blind-16 original within 1.5 mm | 77.0% | 80.7% | 80.7% |
| Best all-160 original within 1.5 mm | 81.5% | 85.6% | 85.2% |

The predeclared treatment-versus-control gate **failed**. The plan-equal
selected error improved only **0.078 mm**, below the required 0.30 mm; the
selected-within-1.5-mm fraction did not improve, against the required ten
percentage points (section weighted, both selected 66/243). Six of eight plans
were non-worse, and rigid error, original-candidate capture, raw-input and
weak-real guardrails passed. The 80.7%-available versus 27.2%-selected blind
gap remains the dominant synthetic failure: a near blind candidate was
available in 196/243 sections, but both arms selected one in only 66. The
direct prior's top-one mapped
error was 3.291 mm for control versus 3.302 mm for treatment, so the direct
pose distribution did not improve either.

The selected branch was identical between arms on **235/243** sections; the
branch and original/corrected action were identical on **219/243**. On those
219, mean mapped-tissue improvement was only **0.0004 mm**. Eight branch changes
accounted for almost all the mean gain (seven improved, one worsened). Raw
no-brush cases gained only 0.043 mm; the 30–45° and 45–55° nearest-cardinal
angle strata gained 0.039 and 0.013 mm. These conditional synthetic strata
are descriptive, not a physical-angle validation.

On 151 support-matched near-versus-wrong pairs, treatment chose the near branch
69.5% of the time, versus 68.4% for control. Its fixed-pair win rate remained
68.9% with atlas intensity zeroed, 66.9% with the source image features
swapped, and 68.2% with visual features removed while candidate position and
support were retained. None met the required ten-point anatomy-reliance
advantage. These are out-of-distribution conditional ablations, not proof of
complete image independence, but they give no warrant to treat the learned
attention as a trustworthy fitting-difficulty signal. On these same treatment
pairs, the image-free direct prior alone won **68.9%**; prior-only and full
matcher decisions agreed on **148/151** pairs. The extra fitting pathway
rarely changed which plausible plane won. The ablations also leave the
original prior score unchanged, so their small gaps are only conditional
evidence tests, not an exact measure of visual causality.

The weak Allen-affine, donor-equal DEV discrepancies were essentially matched
between control and treatment: coronal **0.8282/0.8284 mm** (six donors) and
sagittal **1.4930/1.4900 mm** (eight donors). The references are inherited
automated alignments, not blinded expert truth. The synthetic source still
derives anatomy from one Allen template with procedural deformation and
appearance, and no independently referenced physical steep-oblique cohort is
available. No calibration, final animals, public DeepSlice benchmark or GUI
promotion occurred.

**Decision:** retain the checkpoints as negative development evidence; do not
scale this correspondence loss or claim fitting-to-pose feedback. The next
model change must first make fitted image–atlas agreement discriminate
support-matched correct versus plausible wrong planes *using anatomy rather
than plane position/support*. The previous raw fit difficulty was inverted on
12/15 such pairs, so simply increasing its weight is not justified. Only after
that signal passes source/atlas/geometry ablations should it teach the direct
probabilistic pose head in a matched feedback comparison. In parallel, obtain
physical oblique/DV sections with animal IDs and independent pose/landmark
references; synthetic success alone cannot qualify arbitrary-cut use.

Frozen artifacts: `I:/AnatomyTracker/runs/joint_in_path_correspondence_128_control`,
`I:/AnatomyTracker/runs/joint_in_path_correspondence_128_treatment`, and
`I:/AnatomyTracker/runs/joint_in_path_correspondence_128_dev_eval`. Their
completion-receipt SHA-256 values are respectively
`6f73956ed6333afc660e69788b1392ba40b0961b59ae358bd76d3450ff6c16df`,
`33d3ceca41c31fa302a9ab3796e3cca4ce4ae571294f002f067160eb2f5836ea`,
and `9d05f2e56187125d6d0b9ad7b6c16fa0ba1292504c63a1c11eb31420d15d755c`.
The evaluator independently rechecked source, protocol, panel, TRAIN manifest,
checkpoint and split hashes before producing its summary. A post-exit audit
rehashed every declared evaluator output, all four checkpoints and the logs
from both arms; all matched their frozen receipts, with exactly 6,000 logged
batches per arm, 243 synthetic DEV rows and 222 weak-real DEV rows. An
independent read-only audit additionally matched every TRAIN update/slot and
weak-real identity to the shared manifest, all 256 panel files to their
recorded hashes, and the raw-row gate arithmetic.
