# Preregistered study protocol

## Status and scope

This document specifies the study before the proposed unified model is trained. It does not claim that the architecture, datasets, benchmarks or results described as planned already exist. The implementation baseline is commit `c6681039e0b7acf35c9cdbee43040a3dca29cdab` (`c668103`). Existing evidence is transcribed in [`BASELINE_LEDGER.md`](BASELINE_LEDGER.md); future values must be appended with immutable run and artifact hashes.

The study asks whether one jointly trained model can estimate any
brain-intersecting section's full 3-D Allen CCF frame and 2-D nonlinear
anatomical correspondence more accurately and robustly than the present
two-stage pipeline and applicable published alternatives, while remaining
practical in the desktop application. Recurrent refinement remains a tested
hypothesis, not an architectural commitment before premise evidence.

## Prespecified claims

1. **Pose:** the joint model improves atlas-plane localization over the frozen AtlasPose pipeline and leading automatic comparators.
2. **Dense correspondence:** given the same plane, it improves visible-tissue anatomical correspondence over the frozen dense model and registered classical/learned baselines without increasing folding.
3. **Joint feedback:** recurrent pose--warp feedback improves the full pipeline beyond a feed-forward cascade; deformation does not merely conceal pose error.
4. **Sequence inference:** optional shared-normal and ordered-stack-offset evidence improves series alignment while satisfying every declared hard constraint.
5. **Scientific utility:** improved registration reduces 3-D probe-trajectory and recording-site region-assignment error.
6. **Usability:** ten ordinary whole-brain sections complete within 180 seconds on the reference workstation. This is a ballpark usability ceiling, not permission to trade away accuracy.

Claims 1--5 require locked evidence. Attractive overlays, development cohorts and synthetic results alone cannot establish them.

## Analysis modes

Results are reported separately for:

- **automatic:** image input only, including an automatic visible-tissue mask if used;
- **outline-assisted:** an optional user smart-brush visible-tissue/brain-surface mask, with interaction and outline quality disclosed;
- **constraint-assisted:** optional valid plane domain, stack ordering, shared normal or surgical constraints, with each information source disclosed;
- **expert-assisted:** manual QuickNII/VisuAlign or equivalent workflow.

An assisted result may not be presented as a fully automatic comparison. The primary automatic benchmark gives every method the same raw image and no information unavailable to its comparator.

## Evaluation tracks

### A. Published DeepSlice reproduction

Use the DeepSlice Ground Truth dataset (DOI `10.25949/22802411`) and its
original brain-level development/test assignment; the final public test
component contains only four brains and is therefore comparator reproduction
evidence, not a broad biological or arbitrary-plane holdout. Run the frozen
publication DeepSlice implementation and official recommended workflow. Report
raw single-section and series-assisted outputs separately. Preserve raw raster
bytes and raw-frame ground truth. A new hash-bound adapter applies exactly one
horizontal image reparameterization into its expected A-to-P view and, only
after all official postprocessing, applies the dimension-aware inverse
`O'=O+((W-1)/W)U, U'=-U` for a `W`-pixel raster. Use
the separately versioned QuickNII/webnutil `x/W,y/H` corresponding-pixel
physical plane distance over reference brain. The frozen earlier `+0.5` metric
is reported, if needed, only as a labelled legacy diagnostic.

This public dataset and the local 1,400-section comparison cohort are consumed. They may reproduce a named comparator and diagnose regressions, but cannot be used for hyperparameter selection or a new final-generalization claim. The previously tracked local comparison used the wrong horizontal view for DeepSlice and is invalid until the corrected official-coordinate adapter is frozen and rerun.

### B. New hidden real pose benchmark

Target 30--50 brains, at least three laboratories, several
acquisition/staining conditions and approximately 500--1,000 whole sections.
Recruitment deliberately spans coronal, sagittal, horizontal and
extreme-oblique cutting rather than assuming a coronal convenience sample. The
final sample size is set by a preregistered pilot-based power calculation.
Split strictly by animal and laboratory, never by experiment or slice. Three or more
blinded neuroanatomists create QuickNII-compatible alignments. The consensus
and uncertainty protocol is frozen before model evaluation. A third party
should retain final labels when feasible.

### C. New real dense-registration benchmark

Target 150--250 sections from at least 20 animals. Multiple blinded experts identify 15--30 homologous landmarks and a prespecified set of reliable boundaries/regions per section. Points may be marked unidentifiable; missing tissue is never treated as a correspondence target. Point consensus uses a robust geometric estimator with adjudication defined before evaluation; mask consensus uses the frozen annotation rule. Leave-one-rater-out human error defines the expert variability envelope.

### D. Exact synthetic out-of-generator benchmark

Use exact arbitrary-plane O/U/V, structured frames, forward/inverse maps,
labels and validity masks. Test transformations and artifact implementations
must be independent of the training generator, not merely use new random
seeds. Stratify equal-area orientation cells, offset/support, cardinal and
extreme-oblique planes, clean through severe artifacts, and difficult
neighbouring-plane negatives. This track tests correctness and controlled
robustness, not real-histology validity.

### E. Downstream probe mapping

Create exact CCF digital probe phantoms covering pose, tissue deformation, missing observations and channel geometry. Evaluate an independent real Neuropixels cohort using blinded expert reconstruction and, where obtainable, block-face or cleared-brain reference. Surgical plans are priors, not ground truth.

## Data separation and benchmark custody

- Every real-data split unit is an animal. All experiments, images, augmented descendants and serial neighbours remain within that animal's split; a record without a resolvable animal identifier is ineligible for confirmatory claims.
- Development uses training and validation only. Test data are evaluated after architecture and thresholds are frozen.
- The new hidden real benchmark is consumed once for the release decision. Any subsequent access is labelled post hoc.
- Public DeepSlice data and previously viewed session 722 are development/diagnostic evidence only.
- Each manifest, image tree, annotation file, comparator container, checkpoint, evaluator and result table receives SHA-256 provenance.
- A benchmark-custody record identifies who could see hidden labels and when.

## Primary endpoints

The co-primary endpoints are:

1. brain-level mean DeepSlice physical plane distance on the hidden real pose benchmark;
2. brain-level median landmark target-registration error in micrometres on the hidden real dense benchmark;
3. exact synthetic visible-tissue Allen-label correspondence accompanied by macro regional Dice and topology constraints;
4. per-recording-site 3-D CCF error and hierarchy-aware region accuracy on exact probe phantoms.

Superiority to a comparator requires a paired brain-level confidence interval excluding no improvement on the declared endpoint and no prespecified safety/robustness regression. A model is not declared best on the market unless it beats every locked primary automatic comparator on its applicable primary endpoint. Missing or irreproducible comparators are reported as such, not treated as defeated.

## Secondary endpoints

- physical corresponding-plane error, geodesic normal error, physical offset
  error and in-plane frame error, with legacy AP/L--R/D--V summaries only for
  the coronal comparator subset;
- plane-anchor/corner error and catastrophic-error rates;
- landmark p95 error, macro Dice, bottom-30% Dice, boundary F1, ASSD and HD95;
- forward/reverse endpoint error, cycle error, negative-Jacobian fraction, minimum determinant and SD log-Jacobian;
- risk--coverage and failure-detection AUROC/AUPRC from the compatibility-derived risk ordering;
- proper scoring rules plus 50/80/90/95% empirical coverage and spatial-volume
  width for the pose posterior, including the prespecified check that nominal
  90% credible regions contain approximately 90% of unseen-animal references;
- trajectory entry, angular, roll, depth and region-sequence errors;
- latency, peak memory, failure rate and manual interaction time.

All endpoints are stratified by equal-area orientation cell, physical offset,
visible support, cardinal family, artifact severity, laboratory,
stain/acquisition, damage and distance to an Allen boundary where sample size
permits.

## Statistical analysis

- The animal is the inferential unit. Experiment- and slice-level pooling is descriptive only.
- Comparisons are paired. Confidence intervals use hierarchical bootstrap resampling animals first and slices within animal second.
- A mixed-effects sensitivity analysis includes method as a fixed effect and animal/laboratory as random effects.
- Primary endpoints use two-sided 95% confidence intervals. Secondary multiplicity is controlled with Holm correction within metric families.
- Report absolute effects, relative effects, confidence intervals and raw per-case outputs; do not rely on p-values alone.
- Human noninferiority margins are defined from leave-one-rater-out variability before the hidden model evaluation.
- At least three independent training seeds are evaluated; the model-selection rule is fixed before test access.

## Missingness and failure

Every attempted case remains in the denominator. Crashes, invalid transforms, timeouts, empty masks and non-finite outputs are failures. Unidentifiable expert landmarks are excluded only under the frozen annotation rule. Abstention is allowed only when reported through risk--coverage analysis; an abstained difficult case is not silently removed.

## Stopping and model selection

Training may stop for a frozen early-stopping rule, numerical failure, exhausted compute budget or a futility decision based only on development data. Candidate choice must not inspect hidden test results. The release decision follows [`../../publication/gates.yaml`](../../publication/gates.yaml); failed candidates and negative results remain in the run ledger.

## Reporting

The final report includes architecture/configuration, parameter count, training views, seeds, compute, energy/runtime where available, all data provenance, inclusion/exclusion flow, per-case predictions, confidence intervals, failure gallery, ablations, constraint audit, export parity and limitations. Manuscript text must distinguish measured results from hypotheses and must not convert a passed internal threshold into an unsupported clinical or market-wide claim.
