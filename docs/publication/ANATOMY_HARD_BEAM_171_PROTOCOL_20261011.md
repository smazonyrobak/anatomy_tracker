# 171: hard-beam anatomy-feedback diagnostic (TRAIN only)

The 169 matcher learned same-support intensity contrasts, but on 170's fresh
blind-16 beam its selected initial rigid error was 2.922 mm versus 2.516 mm
for the frozen pose prior (58 nonblank sections). The beam oracle was 1.116 mm.
This short experiment asks whether training on the **actual model-proposed
alternatives**, rather than perturbations of the true plane, closes that gap.
It is a diagnostic, not a final joint registration model.

Freeze the exact random-initialized-only 132 step-2000 pose/map parent and
initialize the 168 five-depth correspondence matcher from 169 step 6000.
Draw each TRAIN section independently from bases 0–59: 80% support-intersecting
166 geometry, 20% full-box 151 geometry, and independent v4 appearance and
artifacts. Do not deliberately duplicate a plane with a new background.
For every section, generate the unmodified 121 blind-16 candidate beam from
the frozen pose model before reading its plane truth. Truth may then select
the beam's best plane, high-prior wrong plane, current anatomy scorer's
highest-ranked wrong plane and, if needed, a near wrong alternative for a
four-way listwise ranking loss. All 16 anatomy scores used for mining are
computed without gradients before truth is read. Each selected candidate
receives one bounded correction and a finite-thickness atlas re-render. The
loss also retains 169's same-support, matched-histogram wrong-intensity
counterfactual and a small best-candidate correction term. No truth, animal
identity or observed valid-target label enters the matcher input or any
inference score; rendered candidate atlas support does enter the score. This
diagnostic supplies the *true synthetic* slab offsets and weights to the
fitter. Unknown slab thickness/PSF needs a separate later ablation.

Before optimizer updates, freeze 128 independent attempts from bases 60–63
using a disjoint seed: 96 unconditional draws and 32 from the conditional
45–55° nearest-cardinal-normal stratum. The latter selects a geometry seed
using only its sampled orientation, before rendering or seeing any tissue,
model score or error; record the number of orientation seeds drawn. Keep
blanks in both strata; never replenish. Report unconditional and heavy-oblique
strata separately, not only their deliberately enriched aggregate. Save exact images, branch
IDs, initial states, reflections, slab parameters, frozen prior/native scores,
truth for evaluation only and full provenance in a hashed panel artifact. Use
the same panel at initialization and at 400, 800 and 1600 optimizer updates.
No checkpoint is selected on this panel. Compare prior, native 121, raw
anatomy, prior-plus-anatomy (fixed 0.02 score temperature), and the matched
score-stage zero-atlas-intensity controls. The null after correction retains
the **actual-intensity-induced** corrected state; it tests scoring, not a
fully atlas-free fitting pipeline. Record per-candidate initial/fitted errors,
atlas support, reflection, all scores and each selected slot, plus per-case
section/specimen/experiment/animal IDs and angle/appearance strata.

This panel is synthetic TRAIN-inner validation, not independent-animal or
acquired-section validation. A useful signal should beat the prior and native
rankers on paired blind selection and the matched intensity-null, including
sparse and oblique strata; a hard-beam loss alone cannot prove that the fit
energy represents coherent deformation. The present 168 score independently
picks local peaks and has no spatially coherent deformation/strain cost.
VoxelMorph's similarity-plus-smoothness formulation
(https://arxiv.org/abs/1802.02604) and iterative correspondence approaches
(https://arxiv.org/abs/2107.10004) are relevant to that later step. Do not
promote to GUI, calibrate probabilities, consume animal-level final tests or
run a public benchmark from this diagnostic.

The predeclared diagnostic promotion gate applies **only to step 1600 on the
96 unconditional inner attempts**, with blank attempts retained in the
receipt but excluded from error denominators. Fused prior-plus-anatomy
selection must improve paired mean initial rigid error by at least 0.25 mm
versus **each** of the prior-only and native-121 selections, and improve
capture at ≤1.5 mm by at least 10 percentage points versus each. Its selected
anatomy-corrected mean error must be lower than native 121's selected fitted
mean error. Against the matched score-stage intensity-null fusion, it must
improve mean initial error by at least 0.15 mm and capture by at least five
percentage points. Report sparse valid-target and 45–55° strata descriptively as
safety checks, including the separately enriched oblique stratum; their small
counts do not set the gate. If the fixed terminal checkpoint fails, do not
keep scaling this independent local-peak matcher: test a spatially coherent
fit/deformation energy instead. Passing this gate would justify only the next
synthetic joint-feedback experiment, not animal-level qualification, GUI
deployment, calibrated probabilities or benchmark claims.

The bounded preflight checks one gradient-bearing TRAIN draw, one blind-16
score/evaluation draw and one orientation-conditioned inner draw. It does not
validate the full frozen panel or an optimizer/checkpoint round-trip; those
occur only if the reviewed long run is authorized.
