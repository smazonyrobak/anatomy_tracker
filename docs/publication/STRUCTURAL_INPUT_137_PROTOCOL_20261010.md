# 137: matched fixed-structure input test of photometric pose invariance

The frozen 131b draw linked the v4 pose-capture collapse to low exposure; the
short 132 adaptation recovered v4 blind-16 availability but left direct first
choice within 1.5 mm at 73/248 (29.4%) and failed its weak-real coronal-donor
guard. Test whether fixed image-derived structure helps the same randomly
initialized 128 lineage use its available candidates. This is one **treatment**
continuation, not a new model family or a second 132 control run.

Start from the exact 128 treatment step-6,000 checkpoint. Keep architecture,
TRAIN synthetic subjects, coronal/sagittal TRAIN donors, optimizer parameter
groups and initial state, 2,000-update cosine learning-rate schedule, valid-pixel
sampling, direct-pose loss and weights, clipping, frozen modules, and v4/v4/v3
plus coronal/sagittal batch structure from 132. Restore the frozen 132 step-0
NumPy/Torch/CUDA random states and AdamW state; check each accepted **and
rejected** synthetic draw and each weak-real section against the complete 132
draw log, then require byte-identical draw logs. Each synthetic presentation
remains a distinct physical plane with exactly one sampled appearance; do not
create paired recolors. No v5/v6 generator or external pretraining is used.

The sole intervention fills previously zero image-input channels 3 and 4, on
synthetic and acquired images alike, using only observed channel-0 pixels.
Channel 3 is intensity divided by the image's 99th percentile (floor 1e-4,
clipped 0–2); channel 4 is the corresponding 17×17 local mean-subtracted
contrast divided by local standard deviation plus 0.05 (clipped −3–3).
Neither receives a tissue mask, atlas rendering, plane, affine, donor identity,
or target. Set the parent encoder's first-convolution weights on channels 3/4
to exactly zero before update 1. Since all other weights and inputs are
unchanged, the treatment's step-0 direct predictions must equal 128/132 step
0 exactly in mathematical arithmetic. The fixed cues are approximately
invariant to a multiplicative exposure change, not to additive noise,
clipping, stain inversion or lost low-SNR anatomy; that is the intervention's
scientific assumption, not a guaranteed property of real histology.

Freeze checkpoints at 0 and **2,000**, with the terminal checkpoint as
the only primary endpoint. After training has exited, evaluate them on the
already frozen 132 v4 panel (248 informative sections from eight reused
synthetic DEV deformation plans), old v3 panel (243 informative), and the same
six coronal/eight sagittal weak-DEV donors. The existing 132 step-2,000
checkpoint is the matched control. Recompute its predictions only to audit
the new all-160 oracle; require its blind-16/top-one rows, and treatment step-0
rows, to reproduce the frozen 132 evaluator. Use the exact 132 blind beam and
observed-valid native-256 rigid-gauge metric. Report selected mean, direct
first-choice, blind-16 and all-160 **truth-selected diagnostic** availability
at 0.5, 1.0 and 1.5 mm, with plan-equal and section-weighted summaries,
exposure/mode/angle strata, and each weak-real donor's five-point discrepancy.
Oracle availability never selects a candidate or checkpoint.

The terminal treatment passes this **development-only** criterion only if,
against frozen 132 step 2,000 on the same eligible v4 sections, direct
first-choice ≤1.5-mm capture increases by ≥10 percentage points, mean
first-choice error decreases by ≥0.25 mm, and first-choice capture in v4
exposure <0.15 increases by ≥10 points; on v3, neither blind-16 nor
first-choice ≤1.5-mm capture drops by >5 points. Every weak-real donor must
remain within +0.20 mm of the frozen 128 parent five-point mean, the original
132 donor guard. Report the full donor-paired comparison against 132 even if
this rule fails. The 132 control's historical failure is immutable and cannot
be relabeled a pass. Any treatment pass warrants fresh-identity and blinded
animal-separated confirmation, not GUI promotion.

Bind parent, 132 control/evaluator, panels, TRAIN/DEV donors, source, protocol,
draws, checkpoints and raw rows by SHA-256 in completed receipts under `I:`.
No sealed/final animals, expert physical-oblique truth, public benchmark,
calibrated probabilities, fitted-map claim, or joint fit-to-pose feedback is
opened by this experiment. The reused synthetic DEV plans and inherited weak
Allen affines cannot establish biological generalization.
