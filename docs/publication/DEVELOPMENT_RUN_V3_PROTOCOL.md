# Arbitrary-plane V3 development run protocol

This is the predeclared first joint-model development run. It is not a public benchmark, final-test evaluation, or calibrated clinical/scientific performance claim.

## Independence and data roles

- Initialize every parameter from the recorded PyTorch seed. Prior model weights, features, embeddings, predictions, and pseudolabels are forbidden.
- Use only pinned Allen CCF assets and authenticated synthetic rows for training and internal development.
- Preserve exact animal, specimen, experiment, synthetic-animal, section, and synthetic-realization identifiers in every row, training receipt, and raw prediction.
- Keep train and internal-development animals disjoint by component-specific ID prefixes. Public DeepSlice Ground Truth, real laboratory histology, external-validation animals, and final-test animals remain untouched.

## First curriculum

The first fast curriculum covers all brain-intersecting planes with Haar-uniform projective normals, length-uniform support-intersection offsets, uniform roll, exact black exteriors, imperfect smart-brush masks, and unmasked/raw-background inputs. It contains 3,072 identity-deformation pose rows and 2,048 affine-free nonrigid rows, grouped as 16 sections per synthetic animal. The internal development cache contains 384 and 256 rows respectively under disjoint identities.

Coverage is distribution-preserving at the row level: one authenticated normal, roll, and support-chord offset is drawn for each global logical index and is never redrawn because its finite raster has too few tissue pixels. The requested pixel threshold is an identifiability label, not an acceptance gate. Marginal/empty raster cases remain authenticated cache rows, use an explicitly recorded identity/censored realization where necessary, and receive zero point-pose and dense-deformation loss weight rather than a falsely unique target. Downstream appearance, damage, outline, topology, or gauge retries use separate deterministic streams while retaining the exact same parent plane. Every row records the support count, threshold, eligibility bits, and zero/one supervision weights; training reports the identifiable fraction and computes pose metrics on that denominator.

These direct rows are single-centre-plane observations. Their runner PSF is therefore exactly `axial_offsets_um=[0.0]`, `axial_weights=[1.0]`. This zero-thickness curriculum establishes pose capture and pose/deformation identifiability; it must not be presented as finite-thickness training. Finite-thickness subject-deformed rows are a subsequent authenticated stage.

## Atlas and search space

- Read `average_template_25.nrrd` and `annotation_25.nrrd` in F index order and bind their raw SHA-256 hashes and decoder versions.
- Construct a two-channel `float32` AP/DV/ML atlas: template intensity clipped after the fixed in-support transform `(x-9)/(273-9)`, with exact zero outside `annotation != 0`; and a binary support channel.
- Use 384 antipodal normals, 16 support-conditioned offsets per normal, and 16 rolls: 98,304 complete catalogue cells with two raster representations per physical cell.
- Use a 160 by 160 raster and a 12,000 by 12,000 micrometre initial field of view. Continuous recurrent updates recover local normal, offset, roll, in-plane translation, scale, and shear residuals.

## Randomly initialized model and optimization

- Separate histology/atlas stems, a shared encoder, complete probabilistic coarse retrieval, shared-weight recurrent correlation updates, and an affine-free stationary-velocity deformation decoder.
- `feature_channels=32`, `hidden_channels=64`, `top_k=4`, three recurrent refinements, and deformation enabled only after the fixed pose-capture iterations.
- Deterministic uniform sampling over the entire frozen composite cache in both phases. During the first 1,000 applied steps the deformation decoder is frozen; identity and deformed views are both eligible so pose learning cannot assume undeformed tissue.
- AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 4, 512-cell authenticated training banks containing the truth cell, AMP, gradient-norm clipping at 5, and a predeclared 4,000-step target. Check and preserve bounded milestones without changing the frozen target.

## Development gates and later validation

First inspect optimization, honest retrieval, physical landmark error, projective-normal/offset error, full-frame error, deformation error, Jacobians, inverse-cycle consistency, failures, abstentions, and smart-brush mode strata on animal-disjoint internal development rows. Save raw per-row predictions and treat animals as the reporting units. Uncalibrated scores remain explicitly labelled as such.

After the method stabilizes, fit uncertainty only on held-out calibration animals and audit 50/80/90/95% coverage without sacrificing point accuracy. Then follow `FUTURE_VALIDATION_PLAN.md`: untouched final-test animals, DeepSlice Ground Truth DOI 10.25949/22802411, separate real laboratory histology where available, blinded expert references, fair tool comparisons, physical landmark error as the primary endpoint, effect sizes with 95% confidence intervals, and immutable splits/configs/seeds/raw predictions.

All caches, temporary files, checkpoints, reports, feature caches, and predictions for this run live under `I:\AnatomyTracker`; C: is not a development target.

## First zero-thickness pilot result

The first protected execution from commit `de9418411aff31320c42b880943a1782b32d22db` authenticated and froze all 5,120 training rows and all 640 animal-disjoint internal-development rows, then entered GPU optimization. Its ledger is internally consistent through 50 attempts and 40 applied updates, but the process exited nonzero before the 4,000-update target. The public post-training gate correctly rejected the incomplete run, so no inference package, raw development predictions, calibration result, benchmark result, or model candidate was produced.

Read-only deterministic reconstruction from the exact step-40 checkpoint advances five further updates and then reproduces the failure at step 45: the initial probabilistic plane-mixture NLL overflows to positive infinity under AMP while retrieval, refined-plane, landmark, deformation, parameter, and optimizer values remain finite. This is a numerical-stability defect in the initial uncertainty objective, not evidence of corrupt cache rows or a scientific performance result. The frozen run will not be resumed or rewritten. Development must fix the loss, verify finite gradients under autocast, and launch a new receipt-bound pilot before any internal-development evaluation.

Exact paths, receipts, hashes, failure localization, and the negative gate decision are recorded in `arbitrary_plane_v4_zero_thickness_pilot_failure.yaml`.

## First finite-thickness S=9 engineering smoke

The first authenticated finite-thickness smoke ran from exact source commit `89bcac192a4a8d3ac456693e3c759b2c9f53e963`. It froze 24 training rows and 12 internal-development rows, retained every logical and marginal/empty row, and preserved zero overlap across animal, specimen, experiment, section, and synthetic-animal identifiers. Both partitions cover the three input modes evenly: raw/no-brush backgrounds, exact black exteriors, and imperfect smart-brush masks. Every row uses the production finite-boxcar contract with nine axial samples, integer masses `[1,2,2,2,2,2,2,2,1]`, global unit mass, per-row thickness in the declared 25--100 micrometre interval, and axial steps below 12.5 micrometres.

The run reached its exact 24-applied-step target in 32 attempts. Eight AMP overflow attempts were skipped by the authenticated runner while the scale fell from 65,536 to 256; all numeric values in committed reports remained finite. The public v4 loader authenticated the complete cache/report/checkpoint ledger, and the capability-bound export verifier accepted checkpoint SHA-256 `4887c8328cc6ad84ff4be21d3105f32a45489106f05c1e606bcb07938d3bc7cd` under export receipt `77a40efbe66965a1dac6dd213126a1d76c6ae6c174cf1c0c9b73738d1376ff6e`. This is an engineering **PASS** for finite cache-to-training-to-export plumbing.

The initial training-only audit left scientific quality **NOT ESTABLISHED**: on all 24 applied updates the model-selected truth-in-top-k fraction was zero while refinement was truth-forced at one. A first CUDA postrun then exposed a CPU/CUDA mismatch during annotation-dependent metric reconstruction. Commit `017879e8f789ac7862fda7d33c520a894c32629d` fixed and regression-tested that mismatch by replaying saved raw predictions and metric inputs on CPU. The partial output, containing only one raw prediction and an incomplete checkpoint export, was archived under `I:\AnatomyTracker\rejected_launches\finite_v4_smoke_postrun_001_annotation_device_bug_9f9e88a` and was not reused.

The newly generated hardened package passes the independent verifier with package receipt `02de5983fca6f82b56ed0c0c01ec0a79139c9f588098f3003084eb215f5ee7ba` and evaluation receipt `a2e94006b8164d7d47c4706beaf37bbfd6cf53e7f19267fa5e25a966e7731697`. It evaluates and preserves all 12 development rows, four per input mode, across two animal units; all raw predictions are bound to their exact cache rows and per-row S=9 schedules. The independently replayed cache scientific audit at commit `3e39887bda6c6fc77c7bb0b8632958563d7762b9` matches receipt `8e2e5033b003ee41378932e20147c760b8b635426992c1fc584958b8b6a43706` exactly and passes all contractual provenance, no-drop, PSF, supervision, optional-brush, finite-value, and identity-disjointness gates.

Scientific quality is nevertheless a clear **FAIL / NOT ESTABLISHED**. All 12 rows failed and animal-macro top-k recall is zero. Across the two animal units, physical finite-frame landmark error is 11,670.37 micrometres, absolute plane-offset error is 3,591.52 micrometres, projective-normal error is 57.04 degrees, finite-frame rotation error is 129.53 degrees, and foreground Dice is 0.000635. Scores and covariances remain explicitly uncalibrated, with no coverage claim. The engineering path remains **PASS**, but the next gate is improved honest coarse retrieval and a newly trained receipt-bound finite S=9 pilot with materially better all-row animal-disjoint pose and overlap results before calibration or public benchmarking. Exact evidence is recorded in `arbitrary_plane_finite_v4_smoke_result.yaml`.
