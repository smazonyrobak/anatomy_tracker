# 309 retrospective: the existing atlas-fit score does not fix selection

This is a **descriptive TRAIN-inner replay**, not a fresh validation panel or a promotion gate. After seeing 309's failed result, we took its first 64 frozen v10 cases (eight from each synthetic base 56–63). The diagnostic regenerated each case from its recorded seed and verified its input, pose and sampled CCF targets against the saved panel bitwise. It kept the batch-20,000 model, prior-top-16 proposals and fitted maps fixed, then applied the already implemented 305 full-intensity, support-only and support-preserving shuffled-intensity scores. All work and inputs were on `I:`; no model update, new draw, reserved animal or public benchmark was used. The remaining 192 v10 cases were deliberately not replayed once the small balanced panel answered this retrospective question.

| Same 64 cases, selected error in mm | Rigid surviving sites | Fitted mapped sites |
| --- | ---: | ---: |
| Direct probability head | **2.322** | **2.319** |
| Existing atlas-intensity fit score | 2.420 | 2.416 |
| Same score, atlas support only | 2.318 | 2.315 |
| Same score, shuffled atlas intensity | 2.343 | 2.340 |

The rigid truth-oracle among these top 16 was 0.964 mm; it is unavailable to inference. Full-intensity ranking changed 17/64 direct choices: ten improved and seven worsened, but the larger harms increased mean rigid error by 0.098 mm. Each arm selected four cases above 5 mm; full intensity introduced one new >5-mm error and corrected one. Its incremental evidence preferred the near plane in 34/53 support-matched near/far pairs, versus 32/53 for support only and 33/53 for shuffled intensity. Pairwise preference did not translate into safer all-candidate choice.

The frozen source-reliability map still separated generator-valid tissue from background on this panel (mean predicted reliability 0.935 versus 0.047 at batch 20,000; 0.940 versus 0.050 at batch zero). On the first 16 cases, true-pose fitted-map error was 0.09967 versus 0.09960 mm at those checkpoints, so there was no gross map degradation from the pose-focused encoder continuation in that narrow check. Neither observation proves reliable tissue masking or deformation on acquired histology.

**Decision:** do not run the existing 305 spatial fit on the full 309 panel, train another copy of the failed atlas ranker, or extend 309's exposure recipe. This result rejects only the fixed score's usefulness on this inspected checkpoint and distribution, not the existence of anatomical information in the images. Future work must demonstrate a practically observable, tail-safe all-candidate signal and a direct first-choice gain on *new* sections before buying a long joint training run. Exact local deformation, physical arbitrary-oblique animal validation, calibrated electrode-region probabilities, GUI replacement and the fair DeepSlice benchmark remain open.

Read-only script: `training/diagnose_first_choice_309_spatial_fit.py`; parent frozen run: `I:/AnatomyTracker/runs/first_choice_exposure_309_001`, completion SHA-256 `8215d55a1bb8371d5b3bb8af2c2ee986dda6ed8c1601ecbe06dfdd1a88ea26eb`. The script's post-replay edit changed only its file-hash reading to streaming and panel loading to `weights_only=True`; the latter load was checked separately. No replay output was frozen as a new qualification artifact.
