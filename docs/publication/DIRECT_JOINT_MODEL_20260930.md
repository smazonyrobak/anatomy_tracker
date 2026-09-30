# Direct coordinate prediction and anatomical fitting — 2026-09-30

## Objective and priority

Ship one standalone, randomly initialized model in the existing desktop Anatomy
Tracker. It must locate arbitrary-plane mouse-brain sections and fit individual
anatomy, use optional surgical/section-range information correctly, and eventually
provide calibrated joint uncertainty for electrode trajectories and region
assignments. No inherited model weights, features or pseudolabels. All work stays
on I:. Use native goal pursuit, not heartbeat automations. The native goal record
was absent on this continuation and was recreated explicitly at the user's request.

The user's central clarification is a **training relationship**: predict pose,
extract the corresponding atlas section, fit tissue, and let anatomical agreement
and deformation difficulty improve the coordinate predictor's future predictions.
Reranking a catalogue or refining the same image without that learning connection
does not fulfill it. Build the intended deployable architecture now and train its
parts in stages; do not build a disposable baseline first.

## Completed experiments closed

Both processes had exited before inspection. `close_joint_v6_final_diagnostics.py`
checked all 11/30 recorded output hashes and reconstructed the following results
from raw arrays; no mismatches. Receipt:
`I:/AnatomyTracker/runs/joint_v6_final_diagnostics_closeout_001.json`.
This compact closeout does not independently rerun either model or solver.

| TRAIN diagnostic, 144 paired trajectories | Initial | Learned update | Feature-Jacobian solve |
|---|---:|---:|---:|
| Mean normal error, degrees | 4.000 | 4.162 | 3.140 |
| Mean centre-surface coordinate error, micrometres | 482.465 | 482.413 | 333.288 |
| Half squared feature cost | .319801 | .319355 | .295797 |

Eight synthetic subjects, three input presentations, six single-axis starts and
three updates. No nonfinite solves. Fixed oracle deformation, correct reflection
and known PSF make this an off-policy favorable local experiment, not global or
biological validation. It implicates the evidence/readout route; the solver sees
vector features while the old updater received scalar costs, so this is not proof
that recurrent weights alone are responsible. Completion SHA-256:
`36c934ca1ffc6c53deadc14ee0cc84b8e5b065247f15298f055cef6c73c8057d`.

The 1,920-update canonical adaptation improved coherent TRAIN normal error
49.297→32.570 degrees and top32 capture .2467→.8775, but old synthetic top32
capture fell .9147→.8705. All three old input-mode capture retention gates failed.
Real TRAIN normal error was 7.093→8.548 degrees. Overall predeclared gate false;
no promotion or threshold waiver. Completion SHA-256:
`13da3dd212e333826458c90bf4ef1cbee27ed92ff5b62f96140e24140569f124`.

## Architecture implemented for the next whole-model lineage

`arbitrary_plane_joint_model_v7.py` contains the direct model, with no catalogue
or checkpoint dependency. A four-resolution residual encoder preserves spatial
information. Eight continuous components predict full finite-frame position,
proper rotation, field of view/shear and reflection probabilities. Normalized
Gaussian factors cover center/log-basis coordinates; an isotropic matrix-Fisher
factor covers each component's full orientation on SO(3), normalized by explicit
one-dimensional Haar quadrature. The mixture, not a single global Fisher mode,
represents separated possibilities. Rotation means use continuous six-dimensional
codes. These distributions have not been calibrated.

A spatial decoder predicts atlas-like contrast. Its synthetic supervision prevents
the fitting metric from being an unconstrained learned scalar. The fitter compares
source and freshly rendered atlas features using local correlation and a shared
recurrent update. It predicts a three-dimensional curved section and through-plane
director, not a full reconstruction of the individual brain. Finite-thickness
rendering integrates the explicit normalized PSF. Mean, slope and affine movement
are removed from the deformation; physical derivative bounds constrain folding.
The fitter does not secretly correct the coordinate predictor's global pose.

The fitting objective combines anatomical agreement, deformation magnitude and
requested physical strain (including before limiting). It differentiates through
atlas coordinates and the fitter into the direct pose head. Dense synthetic
physical correspondence also anchors training. Merely needing little deformation
is not accepted as a good match when anatomy disagrees.

A local low-rank covariance head jointly represents pose and coarse deformation
coefficients; its shared latent is intended to preserve correlation along a probe.
It is trained only in a meaningful local chart. It is not yet a validated credible
volume or an electrode-region probability. Optional brush/mark inputs and explicit
missingness/context slots are in the target network. Surgical constraints still
need training and physical multi-section integration; accepting a context tensor
does not establish effective constraint support. Probe angle is not plane tilt.

## First training stage (same architecture, not the final experiment)

`train_joint_v7.py`: random initialization, seed 2026093001, 6,000 AdamW updates,
batch eight, uniform synthetic TRAIN subject then eligible observation. Existing
512 TRAIN sections from eight synthetic subjects; 128 development sections from
four separate synthetic subjects. Synthetic subjects from one atlas are not
independent biological animals. Three existing raw/black/imperfect-brush versions
are retained with exact section/animal/experiment identities and input hashes.
Only the existing 96-pixel data are used in this stage; upsampling is not treated
as new anatomical information. Higher-resolution generation/training is still
required, using this same multiscale architecture.

First 2,000 updates jointly initialize direct pose prediction, contrast decoding
and fitting around perturbed reference poses. The remaining 4,000 fit the direct
predictions themselves with continuous fitting feedback. Training uses the
reference to select the matching mixture component/reflection for supervision;
deployment has neither and evaluates predicted components. The first connected
update must show a finite nonzero fitting-only gradient at the pose head.
Development outputs are saved at initialization and each 2,000 updates, with all
rows retained and eligible subject summaries. This is a direct-pose developmental
readout, not a full fitted-model qualification. Save whole optimizer/RNG state,
source, schedule and raw predictions; no mixing independently trained branches.

## First joint initialization result

The 6,000-step run completed successfully in about20 minutes, from random weights.
At update2,001 fitting loss alone produced a finite nonzero pose-head gradient
(6.22668). This establishes the connection, not its accuracy benefit.
Post-exit diagnosis `runs/joint_v7_direct_joint_001_learning_diagnosis` preserves
all eligible TRAIN/DEV direct predictions and32 known-pose fitter comparisons.

Eligible synthetic-subject macro normal errors were TRAIN14.87deg versus DEV56.12deg;
centre errors733um versus2,900um. Truth-density-selected TRAIN modes had8.59deg error,
but even the lowest-angle DEV mode among8 averaged29.81deg. Ranking alone cannot
repair the generalization problem. The922 eligible TRAIN presentations represent
only311 physical sections, versus91 eligible DEV sections. These are synthetic
subjects of one atlas, not biological animals.

Known-correct DEV plane mappings had55.67um error without deformation and96.84um
after fitting. Correct-pose fit scores beat controlled wrong-pose scores in31/32
cases, but appearance-score improvement did not imply spatial improvement.
One fixed TRAIN batch had span/shear-NLL head-gradient norm1166, centre319.5,
rotation115.1 and dense-coordinate83.0. Concentrations9.6–20.0 and finite6D frame
vector norms do not support orientation-uncertainty collapse at the final step.
Prioritize distinct TRAIN geometry/appearance, real training inputs, direct physical
pose supervision independent of predicted variance, and known-field fitter anchors.
Preserve the complete checkpoint; no new architecture or public benchmark is justified.

Native evaluation of all16 pose/reflection combinations also completed (448
observations,323.6s). Eligible synthetic DEV macro error was56.12deg and4,601um
over visible coordinate grids; six real DEV donors averaged57.38deg and7,474um
against weak full-canvas Allen affines. Real inputs here were the earlier96px
red-channel crop, not the new GUI-consistent192px collection. Do not call either
coordinate metric blinded physical landmark error. Exact native arrays are saved
in `runs/joint_v7_direct_joint_001_native_development_002`; this is a failed
initial generalization result, not a benchmark score or shipping checkpoint.

## Necessary work after initialization

Parallel desktop integration now has a bounded-memory all-mode inference adapter,
an exact raw-display-to-model affine, optional brush preprocessing matching the
existing three training input channels, and native curved-surface point mapping.
The GUI can install this result and save/load its arrays in session version3
(versions1/2 remain readable). A separate experimental tab now selects a complete
checkpoint, section thickness and CPU/CUDA, and runs a bounded background worker
with no legacy fallback. It explicitly labels accuracy/uncertainty unqualified and
constraints unused. No new default is enabled. One actual frozen TRAIN section
reproduced all three input modes exactly; native point lifting, including a display
flip, matched its stored reference coordinates exactly. These are coordinate
contract checks, not trained-model accuracy or a complete desktop workflow test.
Real-image photometry, trained constraints, confidence calibration and distributable
packaging remain unfinished. No live training artifacts were opened for this work.

The actual experimental button-to-worker-to-install path subsequently passed an
offscreen Qt check on one real TRAIN image with the whole6,000-step checkpoint:
2.313s CPU inference/install,13 native arrays and curved surface restored exactly,
zero coordinate drift for two engineering-only marked pixels. This exposed and
fixed a stale canvas-size bug in archive overlay reconstruction. Receipt:
`runs/joint_v7_gui_workflow_smoke_002/completed.json`. This is execution/persistence
evidence, not anatomical accuracy or visible-desktop validation.

The192px real collection is also complete:1,280 images/58 TRAIN donors and64/6DEV,
using actual downloaded JPEGs, the exact GUI channel-average/percentile pipeline
and full-canvas pixel-centre resize. Source affines, IDs and hashes are retained.
These references remain weak Allen labels, with no real deformation truth. The
new synthetic generator is producing4,096 distinct TRAIN planes with explicit
bounded map interpolation and exact sparse finite-PSF coordinates; no DEV split
changes or learned source dependencies.

1. Establish direct prediction learning and useful fitting feedback, using a
   matched disconnected-feedback comparison only when training is stable. Expand
   the existing generator to adequate independent synthetic subjects, anatomical
   detail, realistic acquisition, partial tissue and full plane coverage. Add the
   provenance-preserved real TRAIN donors with appropriately weak geometry labels.
2. Train the same complete model at useful spatial resolution. Train optional
   constraints and missingness; condition on marked tracks and shared animal-level
   entry/direction rather than multiplying surgical metadata per section. Validate
   unconstrained and constrained accuracy, including contradictory constraints.
3. Calibrate on separate animals. Check physical error, multimodal coverage,
   interval coverage, failure detection, and point-estimate accuracy. Propagate
   correlated pose/deformation and annotation/surgical uncertainty through probe
   coordinates. Do not label uncalibrated normalized scores as 90% confidence.
4. Integrate native preprocessing and inference in the existing GUI: preserve the
   raw-to-model affine, optional brush without mandatory automatic segmentation,
   clicked electrode coordinates, constraint units, saved-session identity and
   displayed alternatives/uncertainty. Package the whole weights and atlas for
   actual desktop use; verify practical memory/runtime on the available GPU.
5. Only after strong internal animal-held-out performance, freeze model and
   protocol and compare fairly with DeepSlice and relevant assisted tools. Public
   DeepSlice Ground Truth DOI 10.25949/22802411 has historical exposure and is not
   an untouched final set. Obtain independent real-lab, blinded expert references
   for the final claim; animals are the statistical units. Primary physical
   landmark error, secondary angle/overlap/failures/correction time/runtime,
   effect sizes and 95% confidence intervals. No benchmark-driven retraining.

No full benchmark has been started. No superiority, biological calibration or
finished GUI deliverable is claimed. Prioritize substantial training and useful
Git checkpoints over another series of catalogue-only controls or large audits.

## Scientific precedents and limits

- [Deep Directional Statistics, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Sergey_Prokudin_Deep_Directional_Statistics_ECCV_2018_paper.html):
  probabilistic angular regression; not proof of our full-frame calibration.
- [Probabilistic Orientation Estimation with Matrix Fisher Distributions, NeurIPS 2020](https://papers.nips.cc/paper/2020/file/33cc2b872dfe481abef0f61af181dfcf-Paper.pdf):
  distributions on proper rotations. Our initial within-mode isotropic factor is
  more restrictive; the local joint covariance addresses a different local chart.
- [BA-Net, ICLR 2019](https://openreview.net/pdf?id=B1gabhRcYX):
  differentiable geometric fitting supports end-to-end feature learning. This
  motivates the learning connection, not a claim of an identical histology model.
- [Osechinskiy and Kruggel, 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3335496/):
  joint anatomical alignment and regularized warped sections; classical
  optimization, not evidence that this new neural model already works.
