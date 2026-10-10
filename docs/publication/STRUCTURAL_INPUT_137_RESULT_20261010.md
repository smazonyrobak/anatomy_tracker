# 137: fixed structural cues do not solve plane proposal or selection

The predeclared matched treatment completed 2,000 batches from the frozen 128
parent. Each batch contained two independently drawn v4 synthetic sections,
one independent v3 section, one acquired coronal TRAIN section and one acquired
sagittal TRAIN section. The 6,000 accepted synthetic physical-section IDs were
distinct. Its complete draw log is byte-identical to the 132 control; the
optimizer, random states, loss and schedule were replayed. Only fixed
image-derived input channels 3/4 and their initially zero first-convolution
weights differed. All treatment step-0 synthetic and weak-real predictions
reproduced frozen 132 step 0; the re-evaluated 132 terminal rows reproduced
its frozen evaluator. The training took 584 seconds after startup on the local
RTX 2080 Ti. Three earlier starts stopped before update 1 on a Python
tuple-versus-JSON-list provenance comparison; the corrected, complete run is
the only trained 137 treatment.

The same frozen panels contain 248 informative v4 sections and 243 v3
sections, each from eight reused synthetic development plans. Errors below
are mean 3-D rigid-plane displacement on observed-valid tissue, in mm.
"Blind 16" is the inference-available beam; "all 160" selects by ground
truth only and is a diagnostic upper bound, never a model output.

| Cohort / checkpoint | First-choice mean | First choice ≤1.5 mm | Blind 16 ≤1.5 mm | All 160 ≤0.5 mm |
| --- | ---: | ---: | ---: | ---: |
| v4 / frozen 132 | 3.282 | 73/248 (29.4%) | 179/248 (72.2%) | 35/248 (14.1%) |
| v4 / 137 terminal | 3.273 | 79/248 (31.9%) | 187/248 (75.4%) | 33/248 (13.3%) |
| v3 / frozen 132 | 3.154 | 74/243 (30.5%) | 197/243 (81.1%) | 51/243 (21.0%) |
| v3 / 137 terminal | 3.123 | 72/243 (29.6%) | 194/243 (79.8%) | 55/243 (22.6%) |

On the v4 exposure-below-0.15 stratum, first-choice ≤1.5-mm capture rose
from 19/83 (22.9%) to 26/83 (31.3%), below the required ten-point gain, and
mean first-choice error worsened from 3.531 to 3.606 mm. Plan-equal and
section-weighted aggregates agreed in direction. The primary v4 first-choice
gain was only 2.4 points and 0.009 mm, missing the required ten points and
0.25 mm. v3 retention passed. The acquired weak-reference donor guard failed:
coronal donor 15935 was 0.256 mm worse than the 128 parent, above its 0.20-mm
limit. The coronal donor-equal discrepancy was 0.904 mm for 132 and 0.889 mm
for 137; sagittal was 1.242 and 1.252 mm. These are inherited Allen affines,
not independent expert anatomical truth.

The all-160 diagnostic narrows the failure: on v4, only 35/248 control and
33/248 treatment sections had *any* candidate within 0.5 mm. At 1.5 mm,
all-160 coverage was 200/248 and 207/248, versus blind-16 coverage 179/248
and 187/248. Thus both precise proposal geometry and first-choice selection
remain inadequate. A local atlas fit cannot reliably recover a precise plane
from a bank that often lacks one; merely reranking the existing branches or
adding another brightness-normalization cue is not the next justified step.
This does not demonstrate that arbitrary-plane localization is impossible.

**Decision:** 137 fails its predeclared development gate and is not a promoted
model. Keep unrestricted brain-intersecting planes as the target, but require
better image-conditioned, submillimetre whole-plane proposal capture and a
fit signal with measured anatomy-specific benefit before joint fit-to-pose
training. Use the experimental v6 local-strain source only after real-deformation
plausibility and retention checks. No GUI installation, numerical uncertainty
calibration, physical steep-oblique validation, sealed animal test or public
DeepSlice benchmark was performed.

Frozen training and evaluation are respectively
`I:/AnatomyTracker/runs/structural_input_137` and
`I:/AnatomyTracker/runs/structural_input_137_dev_eval`. Their completed receipt
SHA-256 values are
`bac4efb812b7adae0afd70e646ec0b59809124a050799d52fd56d6b8bd59a9bc`
and `9da6994d3deabe803dfebff1914dffe7cc52bd08a235b5b7dc89a012d7f3be53`;
the terminal checkpoint SHA-256 is
`8074696ea75b27e93fe618e4d4c7ce3d0da977c99a23fe6571d96976dd4c23ef`.
Independent post-exit checks matched every configured output and checkpoint
hash, the control draw hash, all 1,473 synthetic and 666 weak-real raw rows,
and the recomputed first-choice mean/capture in all six panel/arm cells.
