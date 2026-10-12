# 265: blind spatial image–atlas pose capture, bounded pilot

The 260 dense coordinate model predicts almost one constant normal on full-sphere
synthetic DEV (261), even though its centre is usable. The 264 truth-only
preflight found that 64 fixed antipodal normal anchors, three centre offsets
from 260 and an absolute 16×24-mm span reach the normal/centre/**sorted** span
correction box in 214/227 sections. That does not establish raster-axis order,
roll, shear, finite-raster accuracy, blind ranking or anatomy. Earlier 069/071
global scores and 072 one-step correction did not solve blind selection. This
pilot changes the tested mechanism: candidate-specific **spatial** image–atlas
correlations produce both a score and a bounded physical residual, and the
same weights update twice with an exact atlas rerender after each step.

## Frozen inputs and training

Use the project-only 260 batch-3000 checkpoint **solely as a frozen, image-only
centre scaffold**. It is not a proposed second model at deployment. The
normal/roll/frame grid is fixed without true-state information: 64 antipodal
frames × three ±2/0-mm centre offsets along their normal × **both** 16×24 and
24×16-mm raster assignments × eight 45° rolls × two horizontal reflections =
6,144 states. Nothing from the section truth, deformation, synthetic PSF,
validity mask or biological atlas registration can enter the inference bank.
The coarse bank is rendered from 192 unique 32×32-mm square atlas slabs at
96×96, resampled to the 48×48 candidate raster using the exact 192-pixel
centre/reflection chart. Render the blind top 32 again directly from the Allen
25-µm atlas at 96×96. All renders use the inference-available nominal 25-µm
three-point symmetric normalized PSF, **not** the hidden synthetic 25–100-µm
draw. Score the full bank in bounded chunks; no oracle shortlist or
truth-driven support filter is allowed.

The train-only sampler presents 3,000 fresh independent eligible v7 physical
sections, one appearance/artifact/background mode per section, from 64 TRAIN
synthetic deformation bases. Each section's initial bank is built from its
frozen 260 image-predicted centre. Only for training labels, choose four
nearest existing-bank candidates by reflected five-point physical error, four
hard rank-4:64 alternatives, four rank-64:512 alternatives and four wider
alternatives. Never insert the true state or true reflection as a candidate.
Train the random-initialized spatial pair encoder and shared updater with
initial/final candidate physical-error soft ranking targets, physical residual
loss weighted toward initially reachable candidates, and expected selected
physical error. Each of two updates is limited to ≤7.5° local normal tilt,
≤22.5° roll, ≤1.5 mm translation, ≤0.35 log-span per axis and ≤0.15 shear.
The tilt/translation increments fit the nominal 264 normal/centre residual
budget; composed shear can alter singular spans, so measure reach and actual
correction error rather than asserting exact 264 span containment. Report the
*actual* nearest-bank finite-raster five-point error,
not just the 264 sorted-span bound. Retain draw IDs and all source/checkpoint
hashes. No pretrained feature, weight, pseudolabel or automatic segmentation.

Train a second randomly initialized, identically seeded **support-only atlas
control** on the same physical sections and candidate lists, zeroing atlas
intensity in both training and evaluation while retaining atlas support.
This is a separately trained matched control, not a test-time ablation of the
full-intensity model. Use the same checkpoint-selection rule independently for
both arms; compare both at the full-intensity arm's chosen batch as a paired
readout too.

Before training, freeze a small additional DEV panel of new physical-section
draws from the same eight held-out synthetic 248 subject plans, with IDs,
seeds, image hashes and provenance disjoint from the exact 248 sections. This
is a development panel, not new animals or independent biological validation.
Choose among batch-1000/2000/3000 checkpoints by **lower full-intensity
selected valid-pixel rigid error on those new DEV sections**, without using
the 248 readout for checkpoint choice. Then evaluate the chosen checkpoint
once on all 227 eligible frozen 248 sections. Do not tune to those rows.

## Readout and advancement

Measure per-section blind top-32 pre-correction residual reach (normal ≤15°,
centre ≤3 mm, each sorted span log ratio ≤0.7), top-32 best-of-K corrected
five-point and valid-pixel rigid errors, image-selected valid-pixel rigid
error and unoriented normal angle, and the same metrics for the frozen 260
centre/plane baseline and matched support-only control. Also record the full
bank nearest five-point error, candidate atlas-support fractions, inference
seconds and peak GPU memory. Aggregate equally across eight synthetic plans;
show raw/exact-black/imperfect-brush, visible-tissue and cutting-angle strata.
These are synthetic outcomes only, not measured accuracy on animals.

Advance this capture mechanism only if the frozen-248 **full-intensity**
image-selected equal-plan rigid error improves ≥0.5 mm over same-metric 260,
normal angle improves ≥15°, at least 7/8 plans improve, blind top-32
residual-box reach is ≥75%, top-32 corrected best-of-K rigid error ≤1.25 mm,
and the full-intensity selected rigid error beats the separately trained
support-only control by ≥0.25 mm. The additional fresh DEV readout must have
the same *direction* of selected-rigid and normal improvement; no qualification
if it reverses. Failure is diagnostic, not a reason to increase scale or claim
probabilities. No DeepSlice/public benchmark, calibration cohort, GUI swap or
biological test is authorized by this pilot.

If the gate fails but the separately frozen fresh-DEV selected rigid and
normal errors improve materially from batch 1000 through 3000, stop this run
without promotion and preregister a bounded continuation; flat or worsening
fresh DEV is evidence against merely scaling the same mechanism. Do not
reinterpret the reused 248 panel as independent biology.

If this gate passes, consolidate the spatial pair encoder, probabilistic
candidate proposal and shared recurrent residual updater with the existing
local nonrigid tissue-to-atlas map head in **one** model, train end-to-end from
random initialization on project-owned data, and let the post-warp atlas fit
difficulty backpropagate through rerendered pose states. A frozen 260 centre
model plus a scorer is not the requested deliverable. Subsequent biological
animal-level held-out validation, calibration of pose/electrode-region
probabilities and final public benchmarking remain separate gates.

Scientific analogy only: [SLIV-Reg](https://arxiv.org/abs/2410.18683)
reports blind single-slice spatial 2D/3D matching where direct transform
regression struggled, but its CT/MR setting does not establish performance on
mouse histology. [Histological-sequence mapping](https://doi.org/10.3389/fninf.2018.00093)
warns that flexible nonrigid alignment and scalar similarity can favor a
wrong plane; it does not justify treating this pilot's score as calibrated
anatomical truth.
