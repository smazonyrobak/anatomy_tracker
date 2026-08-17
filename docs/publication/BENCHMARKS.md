# Benchmarks, comparators and ablations

## Fair evaluation matrix

Every applicable method is evaluated in three modes:

1. **Pose-only:** atlas-plane output is scored before nonlinear deformation.
2. **Warp-only:** every dense method receives the same exact/reference plane and visible mask.
3. **End-to-end:** each method uses its own predicted plane and complete warp.

Two diagnostic swaps isolate causality: feed each pose output into one frozen reference warper, and feed one exact pose into each warper. This prevents a strong deformation from hiding poor pose and prevents strong pose from hiding poor local correspondence.

## Primary comparators

- **DeepSlice publication release:** frozen official weights/code and its recommended model ensemble, angle integration and cutting-index processing. Compare both raw single-slice and series-assisted modes.
- **AMBIA:** automatic localization plus its documented deformable registration, containerized if reproducible.
- **Xiong et al.:** sequence plane mapping and nonrigid registration when the published implementation can be reproduced on the benchmark contract.
- **Frozen AtlasPose + AtlasWarp:** baseline implementation at commit `c668103` with checksum-pinned models.
- **Expert QuickNII + VisuAlign:** blinded human-assisted operational reference, not labelled automatic.
- **Classical dense registration:** ABBA/Elastix or an Ardent-equivalent initialized from the same plane, when licensing and reproducibility permit.

Comparator identities, revisions, modes and acquisition dates are frozen in [`../../publication/comparators.lock.yaml`](../../publication/comparators.lock.yaml). Failure to reproduce a comparator is reported with the attempted environment and error; it is not scored as a loss.

## Architectural controls

- current pose-only ConvNeXt/AtlasPose;
- current dense-registration model on exact planes;
- VoxelMorph-style pairwise dense registration;
- one strong coarse-to-fine diffeomorphic baseline (LapIRN family);
- one-pass joint pose/warp network;
- recurrent model with stopped registration-to-pose feedback;
- full proposed recurrent model.

All newly trained controls receive identical data manifests, preprocessing, seed count and comparable search budgets. Parameter count, FLOPs where reliable, training compute and inference hardware are reported.

## Metrics

### Pose

- official DeepSlice corresponding-pixel plane distance in CCF voxels and micrometres;
- AP/L--R/D--V MAE, median, p95 and signed bias;
- plane anchor/corner TRE;
- catastrophic errors over frozen physical thresholds;
- per-animal and worst-stratum results.

### Dense correspondence

- landmark TRE median and p95 in pixels and micrometres;
- exact visible-tissue Allen-ID correspondence;
- macro and bottom-30% regional Dice at declared ontology depths;
- boundary F1, ASSD and HD95;
- forward/reverse endpoint and cycle error;
- fold fraction/count, minimum Jacobian and SD log-Jacobian.

### Compatibility-derived risk

- risk--coverage and area under risk--coverage;
- AUROC/AUPRC for exceeding frozen error thresholds;
- accepted-case error at fixed coverage.

The version-1 risk score is evaluated only as an ordering statistic. It is not
reported as a calibrated probability; probability calibration would require a
separately prespecified development-only fit and qualification.

### Probe utility

- cortical entry distance, trajectory angular/roll error and tip/depth error;
- per-site 3-D CCF error;
- exact and hierarchy-aware region accuracy;
- structure-sequence agreement along the shank;
- accuracy stratified by distance to atlas boundaries.

### Runtime and reliability

- warm and cold latency for 1, 2, 4, 8, 10, 16 and 20 slices;
- ten-slice end-to-end p95 below 180 seconds on the reference workstation;
- peak GPU/CPU memory, throughput and provider;
- crash, timeout, invalid-map and abstention rates;
- manual actions and correction time.

## Robustness

Use paired clean/corrupted images so degradation is measured within sample. Sweep rotation, scale, brightness/background, tiling/vignette, tears, missing cortex, occlusion, blur, bubbles, specks and blowout independently and in combinations. Separately hold out laboratory, acquisition/stain and unusual anatomy. Report full severity curves rather than one aggregate.

The independent synthetic benchmark reuses Allen CCF anatomy, and the AtlasPose
warm start was exposed to the full AP domain. Its independent manifests,
deformations, artifacts and dense-training AP blocks test transformation recovery;
they do not constitute unseen anatomy or unseen AP coordinates. Only animal- and
laboratory-held-out real data can support that generalization claim.

## Prespecified ablations

1. frozen two-stage versus unified training;
2. one, two, three and selected higher recurrent iterations;
3. joint feedback versus stopped registration-to-pose gradient;
4. wrong-plane ranking present versus absent;
5. bounded affine-free versus unrestricted local field;
6. CCF synthetic only versus curated real appearance/pose supervision;
7. independent slices versus transparent common-tilt/partial-order solver;
8. automatic/no mask versus outline-assisted mask.

Ablations are evaluated on development data and the frozen final selected set; a large post hoc factorial search is prohibited.

## Reviewer-facing evidence package

- exact reproduction script and raw predictions for the published DeepSlice benchmark;
- new multi-laboratory hidden pose and dense benchmark;
- animal-level confidence intervals and human inter-rater envelope;
- independent synthetic generator and exact probe phantoms;
- public per-case outputs and failure gallery;
- containers/lockfiles, model/data cards and source/checkpoint hashes;
- final benchmark evaluator with hidden labels when feasible;
- independent-laboratory replication as the strongest optional confirmation.

No claim of market-wide superiority is made until all applicable primary comparators run successfully and the frozen superiority rules pass.
