# Training and release protocol

## Prior independent joint model (historical comparator, not a dependency)

The earlier candidate was trained from random initialization as one recurrent
pose-and-deformation model. Its code is retained for audit and possible fair
comparison, but it does not constrain the architecture of the new
arbitrary-plane track and none of its weights, features or pseudo-labels are
used there. It has no learned dependency on AtlasPose, AtlasWarp, DeepSlice or
an ImageNet backbone. The historical implementation is split into the
hash-bound data contract
(`independent_joint_data.py`), model families (`independent_joint_model.py` and
`independent_joint_variants.py`) and audited trainer
(`train_independent_joint.py`). No long architecture-screen result or accuracy
claim is recorded yet.

The source input is grayscale plus an optional tissue-outline channel and an
explicit outline-availability flag. Training mixes 35% accurate outlines on a
black exterior, 35% deliberately imperfect outlines on a black exterior and
30% absent outlines with the acquired background retained. Tissue damage and
dense-validity masks are separate supervision and can never be inferred from
the model-input outline. Accordingly, later evaluation reports automatic
no-user-outline, common automatic-outline and smart-brush-assisted tracks
separately.

Synthetic samples provide exact AP/L--R/D--V pose, similarity, affine-free SVF,
forward/inverse maps and Allen labels. Product 5 provides pose and candidate
ranking supervision only—never invented dense correspondence. Candidate order
is shuffled, recurrence starts from the model initializer rather than the
truth-centred candidate set, and only the true synthetic plane receives the
single dense-registration loss. Checkpoints bind the random initial state,
model graph, all three data streams, atlas assets, optimizer/schedule, animal
IDs, EMA state and raw per-animal predictions.

The pose head emits both point estimates and an uncalibrated full covariance.
Calibration is deliberately deferred until the architecture is frozen, using
animal-disjoint calibration-fit and calibration-check cohorts; until then these
values are distributions produced by the model, not calibrated confidence.

## Arbitrary-plane complete synthetic generator

The arbitrary-plane track now has a deterministic finite Allen CCF renderer
and a complete, standalone G1--G3 synthetic realization. The predeclared
implementation protocol is
[`../publication/arbitrary_plane_synthetic_preflight.yaml`](../publication/arbitrary_plane_synthetic_preflight.yaml).
The development-only result is recorded in
[`../publication/arbitrary_plane_synthetic_smoke.yaml`](../publication/arbitrary_plane_synthetic_smoke.yaml).
The generator adds three stages without using any learned checkpoint,
previous-model weight, pretrained feature extractor or learned style model:

1. **G1 deformation:** two-scale, affine-free stationary velocity fields in
   physical section coordinates, integrated as `exp(v)` and `exp(-v)` by
   adaptive scaling-and-squaring. Ordinary samples must pass positive-Jacobian,
   forward/inverse composition, tissue-coverage, integer-label and exact-replay
   gates. Tiny or tangent cases remain a separately named stress stratum.
2. **G2 appearance:** clean, template-derived, label-conditioned and mixed
   grayscale views with randomized polarity, gamma, gain, offset, low-frequency
   bias, blur, resolution and noise. Label-only views are capped until montage
   review so Allen parcellation edges cannot become a guaranteed shortcut.
3. **G3 observation damage and outline:** tears, missing tissue, holes,
   occlusion and fold-like ambiguity alter the observed image and validity
   masks, never the anatomical deformation. Accurate, deliberately imperfect
   and absent-outline descendants share the same latent sample; the smart brush
   remains optional, and its outline is never the dense-validity target.

The complete generator passed 116 combined arbitrary-plane checks and an
additional 100-seed fixture smoke with no failures. This establishes its narrow
engineering contract, not realism or model performance. All numeric ranges and
development mixtures remain engineering defaults pending a seed-hidden montage
audit. The checks use synthetic train/development fixtures only; they do not
authorize full benchmarking or access to final-test animals. The semantic
arbitrary-plane oracle subsequently passed, but the 64-case model-free
image-information pilot failed its imperfect-outline consistency gate: context
MIND top-1 was 63/64 absent, 62/64 accurate and 52/64 imperfect. The standalone
probabilistic retriever and recurrent joint updater may be implemented,
source-tested and exercised on bounded development pilots, but no
mask-mechanism, calibration, benchmark, qualification or release claim follows
from those pilots. The separately frozen paired mask-mechanism replay and a
positive-Jacobian non-SVF development holdout remain required before broad
model claims, including a check for generator/decoder parameterization
matching.
The model-free oracle protocol is predeclared in
[`../publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml`](../publication/arbitrary_plane_oracle_pose_ranking_preflight.yaml).

The current v3 implementation is a fresh, randomly initialized joint model. It
keeps one canonical continuous state per antipodal physical plane cell,
derives and marginalizes declared raster nuisance representations, normalizes
only a verifier-complete catalogue, renders an explicit finite-thickness PSF,
and rerenders after every shared recurrent pose update. Once pose-only
iterations have captured the large displacement, a shared decoder predicts an
affine-free stationary velocity field in the fixed uniform canvas gauge.
Active recurrent deformation outputs receive exponentially later-weighted
sequence supervision (`gamma=0.8`). The returned point estimate and posterior
summary use the final recurrent state, while the immutable raw-prediction
artifact deliberately retains every recurrent pose/deformation sequence for
audit and failure analysis. The three-coordinate local covariance is
uncalibrated, refinement masses remain conditional within top-K, and exact
omitted retrieval mass is retained separately.

`arbitrary_plane_staged_training.py`, `arbitrary_plane_row_cache_v3.py` and
`arbitrary_plane_training_runner_v3.py` provide the source-bound training and
resume path. They reject learned dependencies and non-development rows, retain
animal/specimen/experiment IDs and exact row/candidate-bank receipts, use two
crash-safe rolling resume slots plus sparse immutable archives, and restrict
generated rows, checkpoints and raw reports to the `I:` development volume. No
pretrained weights, prior-model features or pseudo-labels enter this path.
Checkpoint history is compact: each step retains authenticated row and
candidate-bank hashes in a binding-seeded ordered SHA-256 chain, while the one
full candidate-bank receipt remains in the immutable runner report. Resume and
export reconstruct and cross-check that chain against the reports rather than
duplicating an ever-growing history in every checkpoint.

`arbitrary_plane_pose_curriculum_v3.py` is the fast initial data path. It draws
support-conditioned Haar-uniform antipodal planes and length-uniform valid
offsets from the pinned Allen assets. One pose draw is retained per global
logical sample: finite-raster pixel count is recorded as supervision
identifiability and never causes a normal/offset/roll redraw. Marginal and empty
rasters use an authenticated censored path with zero point-pose/dense loss
weights. Separate deterministic realization retries preserve the same parent
plane. The pose curriculum otherwise forces exact identity G1 deformation,
and retains varied appearance, damage, raw-background, exact-black and
imperfect-brush descendants. It emits the same authenticated v3 row contract
with full animal/specimen/experiment/section lineage and no learned
dependencies. Its focused and row-cache regressions passed 13/13; a separate
authentic Allen 160-by-160 row was generated, frozen, reloaded and audited with
12,531 rendered brain pixels. This is an input-pipeline result, not model
accuracy evidence.

`arbitrary_plane_joint_curriculum_v3.py` adds exact nonidentity, affine-free
multiscale deformation in explicit mild and moderate amplitude bands wherever
the retained raster meets the declared identifiability threshold. Marginal
rows stay in the distribution as explicitly identity/censored observations
rather than being replaced by easier planes. Sampled
similarity is fixed to identity and the uniform-canvas affine SVF component is
moved into the physical pose, preserving pose/deformation identifiability. One
authentic Allen 160-by-160 row had 0.363 px velocity RMS, positive forward and
inverse Jacobian minima (`0.878` and `0.870`) and 0.0312 px gauge recomposition
error. A composite binding places identity-pose and nonidentity-joint rows in
one frozen cache without merging or concealing their component provenance.
These fast rows are explicitly single-centre-plane finite-FOV curricula; they
are not mislabeled as finite-thickness histology. Authentic section thickness
comes from the slower subject-slab path, while the model renderer integrates an
explicit through-plane PSF.

Complete-catalogue inference may use an exact, same-checkpoint atlas-feature
cache. The cache is bound to the checkpoint, atlas, catalogue, PSF, raster,
dtype, layout and build chunking; it covers every representation and cannot be
reused after any binding changes. `arbitrary_plane_development_evaluation_v3.py`
then performs honest complete-catalogue, non-teacher-forced evaluation on an
animal-disjoint development cache, writes immutable raw predictions per row,
reports physical landmark error plus pose/deformation/failure/mode metrics and
labels all uncertainty as uncalibrated. It does not open benchmark or final-test
data.

`arbitrary_plane_acquisition_window_v3.py` is the additive replacement for
the known v2 tissue-centred crop without modifying frozen v2 receipts. Its plan
API accepts only `root_seed`, `split` and `sample_index`; parent and canvas
shapes are fixed by the receipt-bound implementation and pose/tissue never
enter plan sampling. Application records partial/empty views rather than
redrawing.
The window affine is composed into the physical QuickNII plane while its
residual deformation is re-expressed as an affine-free SVF and pullback map on
one fixed uniform canvas. Observation, training-row and cache receipts bind
that gauge, so this window path is part of complete authenticated v3 rows.

One non-frozen authentic pinned-Allen general-oblique integration run completed
the full v3 chain. Its gauge recomposition error was `0.021849987` px against
the receipt-bound `0.05` px maximum (prepared receipt
`e116a8a570133689ae59a2ede764be2f36580da443ac8491c4f586748a123ede`, row
receipt `8ccc91ce47403cf47cb949e0f5a17c4a57e66dbfa52683d72e1788ef4c3e7fa5`).
The run also identified the throughput boundary: authenticated parent
construction took about 23 minutes, whereas one descendant observation took
16.8 seconds. These timings are development evidence only; they require a
content-addressed prepared-parent cache before large generation, and the
ephemeral integration artifact itself is not represented as a frozen dataset.

Future real validation remains strictly animal-split, with untouched final-test
animals. Allen/synthetic data are for development; the DeepSlice Ground Truth
dataset (DOI `10.25949/22802411`) is the public benchmark, and separate real lab
histology is the preferred external validation. Comparisons use blinded expert
reference alignments, physical landmark error as the primary endpoint, and also
report plane/angle error, regional overlap, failures and correction time.
Animals are the statistical units; effect sizes receive 95% confidence
intervals, and exact splits, code, configs, seeds and raw predictions are saved.
Probabilistic pose and dense-map outputs are calibrated only later on unseen
animals, with nominal-coverage and point-accuracy checks.

### Primary sources for the generator protocol

1. Ashburner J. (2007), “A fast diffeomorphic image registration algorithm.” [DOI](https://doi.org/10.1016/j.neuroimage.2007.07.007)
2. Mansi T, Pennec X, Sermesant M, Delingette H, Ayache N. (2011), “iLogDemons: A Demons-Based Registration Algorithm for Tracking Incompressible Elastic Biological Tissues.” [DOI](https://doi.org/10.1007/s11263-010-0405-z)
3. Dalca AV, Balakrishnan G, Guttag J, Sabuncu MR. (2019), “Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces.” [DOI](https://doi.org/10.1016/j.media.2019.07.006)
4. Hoffmann M, Billot B, Greve DN, Iglesias JE, Fischl B, Dalca AV. (2022), “SynthMorph: Learning Contrast-Invariant Registration Without Acquired Images.” [DOI](https://doi.org/10.1109/TMI.2021.3116879)
5. Billot B, Greve DN, Puonti O, Thielscher A, Van Leemput K, Fischl B, Dalca AV, Iglesias JE. (2023), “SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining.” [DOI](https://doi.org/10.1016/j.media.2023.102789)
6. Lee BC, Tward DJ, Mitra PP, Miller MI. (2018), “On variational solutions for whole brain serial-section histology using a Sobolev prior in the computational anatomy random orbit model.” [DOI](https://doi.org/10.1371/journal.pcbi.1006610)
7. Tward D, Brown T, Kageyama Y, Patel J, Hou Z, Mori S, Albert M, Troncoso J, Miller M. (2020), “Diffeomorphic Registration With Intensity Transformation and Missing Data: Application to 3D Digital Pathology of Alzheimer's Disease.” [DOI](https://doi.org/10.3389/fnins.2020.00052)
8. Casamitjana A, Lorenzi M, Ferraris S, Peter L, Modat M, Stevens A, Fischl B, Vercauteren T, Iglesias JE. (2022), “Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas.” [DOI](https://doi.org/10.1016/j.media.2021.102265)
9. Tellez D, Litjens G, Bándi P, Bulten W, Bokhorst J-M, Ciompi F, van der Laak J. (2019), “Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology.” [DOI](https://doi.org/10.1016/j.media.2019.101544)
10. Wang NC, Kaplan J, Lee J, Hodgin J, Udager A, Rao A. (2021), “Stress Testing Pathology Models with Generated Artifacts.” [DOI](https://doi.org/10.4103/jpi.jpi_6_21)
11. Tian L, Greer H, Vialard F-X, Kwitt R, San José Estépar R, Rushmore RJ, Makris N, Bouix S, Niethammer M. (2023), “GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency.” [DOI](https://doi.org/10.1109/CVPR52729.2023.01734)

## Historical AtlasPose protocol

AtlasPose predicts `[AP from bregma (um, anterior positive), L-R tilt (degrees), D-V tilt (degrees)]` from a coronal mouse-brain section. The auxiliary orientation logit resolves the 180-degree ambiguity left after smart-mask roll canonicalization; it is not another anatomical coordinate.

## Deployed v6: historical evidence only

The historical local ConvNeXt-Tiny v6 model was selected on synthetic Allen CCFv3 sections and trained on 100,000 unique synthetic views. Its sealed synthetic-test MAE was 58.72 um AP, 0.934 degrees L-R, and 1.052 degrees D-V.

The 148-section published DeepSlice set was subsequently used as development feedback. It is therefore **not** an untouched holdout and cannot support a production-release claim. On that set, v6 reached 245.20 um, 1.639 degrees, and 3.996 degrees MAE, versus 174.26 um, 1.463 degrees, and 1.268 degrees for the published DeepSlice outputs. The optional 20% AtlasPose / 80% DeepSlice vote reached 140.61 um, 1.20 degrees, and 1.63 degrees. These results explain the current GUI default—DeepSlice—but must not be reused as final evidence for v7.

The v6 model is an experimental predictor, not a demonstrated DeepSlice replacement. It is not source-approved for a new release.

## Historical v7 candidate plan (superseded; not a dependency)

The pending v7 experiment compares the same task-specific pose heads and registered/synthetic data across three ImageNet-initialized backbones:

1. `legacy_xception.tf_in1k` — the closest backbone control to DeepSlice. [DeepSlice](https://www.nature.com/articles/s41467-023-41645-4), [Xception](https://arxiv.org/abs/1610.02357)
2. `convnext_tiny.fb_in22k_ft_in1k` — a modern convolutional backbone with Apache-2.0 pretrained weights. [ConvNeXt](https://arxiv.org/abs/2201.03545)
3. `maxvit_tiny_rw_224.sw_in1k` — a convolution/attention hybrid with Apache-2.0 pretrained weights. [MaxViT](https://arxiv.org/abs/2204.01697)

The non-commercial ConvNeXtV2 FCMAE checkpoint is deliberately excluded. Export provenance records the exact timm identifier, upstream URL or Hugging Face identifier, declared license, and a SHA-256 of the initialized backbone state.

Three pose representations are compared: direct physical regression, physical AP/tilt bins with residuals, and QuickNII OUV regression. All receive the same tolerance-normalized physical-pose objective. The coarse-anatomy decoder is auxiliary training supervision only.

## Data and augmentation

`synthetic_atlas.py` renders exact oblique planes from `average_template_25.nrrd`; `annotation_25.nrrd` provides the known brain support. Training covers AP +500 to -4500 um and L-R/D-V tilt -35 to +35 degrees. Smart-mask preprocessing removes input canvas scale and in-plane roll as nuisance variables.

Synthetic views include the requested range of nonlinear warps, arbitrary rotation, 0.5-1.5x scale, missing tissue and edge-to-edge occlusion, contrast/gamma/offset and polarity changes, non-black backgrounds, local exposure, repeating tile/vignette artifacts, blur/noise, bright specks and blowouts, and artificial tears. Clean and mild views remain present. Fixed seeded manifests make every comparison reproducible and prevent validation/test images from entering training.

Allen product 5 ConnProj serial two-photon (STP) sections are the trusted real-image supervision and checkpoint-selection cohort. Allen product 8 ConnTG slide-mounted affine labels are diagnostic-only: current validation shows specimen-wide registration offsets, and DeepSlice's published curation excluded suspect slide-mounted alignment vectors. Product 8 records therefore cannot affect training loss, model selection, calibration, or a release gate. Dataset manifests, downloaded images, quality exclusions, atlas files, code, dependencies, and pretrained initialization are hashed in the run/export provenance.

The existing 1,400-section, 10-specimen DeepSlice comparison cohort has already been consumed. It is historical regression evidence, not an untouched holdout. Its tracked DeepSlice predictions also used an incorrect image-orientation contract: the raw Allen raster was not transformed into DeepSlice's expected A-to-P view. Consequently the tracked DeepSlice plane-distance, L--R and joint-superiority statistics are invalid and must be regenerated from raw images and immutable raw-frame ground truth with the frozen orientation adapter described below. No existing public or internal DeepSlice cohort may be reused as new final evidence.

The historical local sealed evaluator records useful hashes and failure receipts, but it is not an externally controlled one-shot service. Publication-grade real generalization requires a newly collected specimen-level, preferably multi-laboratory holdout controlled by an independent custodian after model and comparator freezing. For the corrected historical DeepSlice rerun, raw image bytes and recorded ground truth remain unchanged; the comparator adapter applies exactly one deterministic horizontal raster flip, runs official DeepSlice AI/MEns/CI processing, then transforms the final O/U/V plane back to the raw image frame before scoring. This correction is a frozen intended-preprocessing adapter, not per-image assistance.

## Prespecified experiment

- Screen direct, binned, and OUV heads at 20,000 unique views with three training seeds.
- Continue the two best heads to 100,000 views. The winning ConvNeXt-Tiny/head runs are reused rather than trained a second time during the backbone comparison.
- Compare Xception, ConvNeXt-Tiny, and MaxViT-Tiny at 100,000 views with the selected head and the same three seeds.
- Run 20,000-view renderer, consistency, and anatomy ablations against an explicit selected-model control.
- Train the selected configuration on up to 1,000,000 unique views with validation-based early stopping.
- Evaluate the trusted product 5 historical regression split after selection, rerun the consumed DeepSlice cohort with the corrected frozen orientation adapter, and reserve final real claims for a new externally held cohort.

Model-family decisions use trusted product 5 registered-validation animal-level hierarchical bootstraps with a prespecified tie order. Product 8 metrics are reported separately for diagnosis and never enter eligibility or ranking. Smoke runs always report metrics, but cannot pass validation eligibility unless product 5 contains at least 20 independent animals, at least 20 animals in every 500-um AP band, and at least 10 animals in every preregistered L-R and D-V tilt bin. The release gate requires AP MAE and its deterministic animal-bootstrap upper 95% bound <=60 um, L-R MAE <=0.90 degrees, D-V MAE <=1.75 degrees, absolute AP bias <=25 um, AP 95th percentile <=150 um, per-animal and subgroup P90 limits of 90 um/1.50 degrees/2.50 degrees, and worst 500-um AP-band MAE <=90 um. Fixed synthetic validation must also pass overall, artifact-cohort, tilt-band, and paired-appearance invariance gates before a checkpoint is export-eligible. Comparator superiority additionally requires a harmonized rerun with the frozen intended preprocessing for every method and a paired animal-level confidence interval on the declared physical-plane endpoint. The consumed cohort may diagnose regressions but cannot establish final generalization; that claim requires a new external holdout. No v7 superiority is claimed from the existing tracked DeepSlice report.

Passing metrics alone does not deploy a model. The runtime accepts AtlasPose only when the ONNX model, metadata, raw sealed predictions, metrics, presealed commitment, exclusive claim, completion receipt, and release report form one verified hash chain whose model, metadata, and release-evidence hashes are pinned in application source. Promotion emits proposed hashes for human review and never edits those source pins.

## Reproducing the historical v7 plan

Use the CUDA environment and explicit workspace/data roots:

```powershell
$env:PYTHONPATH = "$PWD"
$env:ATLAS_POSE_V7_WORKSPACE = "J:\AtlasPoseTraining_v7"
$env:ATLAS_POSE_REGISTERED = "J:\AtlasPoseTraining_v7\allen_registered_full_quicknii_ras_v2_20260811"
$env:ATLAS_POSE_ATLAS = "$PWD\data\Allen Brain Atlas 25um"
python -m training.train_atlas_pose_v7
```

The selected FP32 ONNX binary is intentionally ignored by Git because it exceeds GitHub's normal 100 MB single-file limit. CPU and every available application accelerator provider must agree with PyTorch within the export tolerances before promotion.

DeepSlice's repository and PyPI metadata disagree about redistribution terms; obtain maintainer clarification before closed-source redistribution. AtlasPose uses Allen-derived data and pretrained timm weights; this project does not itself grant redistribution rights for Allen data or derived weights.
