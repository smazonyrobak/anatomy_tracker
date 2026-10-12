# V9 generator audit 285 result (2026-10-12)

## Outcome

The frozen TRAIN-only audit completed all 512 independent proposals (eight per each of 64 deformation bases) without retries or model updates. There were 473 generator-eligible sections (92.38%) and 512 unique physical-section IDs. Independent recomputation verified every output/source/protocol hash, per-base draw count, within-base variant uniqueness, eligibility count, unit normals, and the affine-decomposition identity.

The consequential finding is a pose-gauge mismatch: on eligible sections, a median 49.51% of squared displacement from the saved plane lies in the support-restricted affine `[1,x,y]` subspace that the mapper removes from its local field. Pixel-pooled across eligible intact sites, the affine fraction is 56.99%. The v9 pose label is the pre-artifact source gauge, not a post-artifact fit. Thus a mapper trained with that pose held fixed cannot represent a large part of its synthetic coordinate target through its affine-free local field.

## Plane and support coverage

| Measure | Proposed (n=512) | Eligible (n=473) |
|---|---:|---:|
| Nearest-cardinal angle q10 / median / q90 | 15.03 / 34.29 / 45.79 deg | 14.68 / 33.96 / 45.70 deg |
| Mean absolute normal components AP / DV / ML | 0.5066 / 0.5109 / 0.4905 | 0.5080 / 0.5111 / 0.4847 |
| Through-support-point branch | 397 | 397 |
| Uniform-box-offset branch | 115 | 76 |
| Full 2,048 valid sites | 456 (89.06%) | 456 (96.41%) |
| At least 128 valid sites | 473 (92.38%) | 473 (100%) |

Eligibility barely changes orientation coverage, and every base contributes six to eight of eight proposals. It does strongly condition the sampling branch: support-point proposals were 397/397 eligible, whereas uniform-box proposals were 76/115 eligible (66.09%). Seventeen eligible sections fell short of 2,048 sites; the eligible minimum was 269 sites. Therefore a `dense2048` request is full-sized for most but not all accepted sections.

## Realized artifacts

"On tissue" is the frozen definition `event && tissue_placed`; it does not infer prevalence in real histology.

| Artifact | Event count / rate over proposals | On-tissue count | On-tissue given event, all proposals | On-tissue given event, eligible |
|---|---:|---:|---:|---:|
| Fragment | 117 / 22.85% | 79 | 67.52% | 73.83% |
| Fold | 102 / 19.92% | 73 | 71.57% | 81.11% |
| Bubble | 64 / 12.50% | 50 | 78.13% | 84.48% |
| Tile seam | 125 / 24.41% | 83 | 66.40% | 74.77% |

The lower all-proposal realization rates are partly the intended no-retry inclusion of low/no-support planes. Among eligible sections the realized rates are close to the configured 80% tissue-placement probability, although fragment and tile seam remain lower in this finite draw.

## Pose gauge and representable displacement

| Eligible-section metric | Median | q90 |
|---|---:|---:|
| Total target displacement RMS | 0.2135 mm | 0.4235 mm |
| Support-affine component RMS | 0.1369 mm | 0.3437 mm |
| Representable local residual RMS | 0.1326 mm | 0.2621 mm |
| Affine fraction of squared displacement | 49.51% | 83.17% |
| Saved-to-robust-fit mean plane separation | 0.1199 mm | 0.3206 mm |
| Saved-to-robust-fit centre separation | 0.0846 mm | 0.3260 mm |
| Saved-to-robust-fit unsigned normal angle | 0.378 deg | 0.919 deg |

The pixel-pooled decomposition is total/affine/local RMS = 0.2598/0.1961/0.1704 mm and 56.99% affine squared error. Its Pythagorean relative error is `1.30e-15`; the maximum per-section OLS orthogonality error is `7.20e-13`, confirming that the split was computed correctly.

This is the valid-intact-support analogue of the mapper's full-canvas projection, not an assertion about unobserved pixels. Even with that limitation, it is large enough to explain why adding dense sites can expose little usable signal when the pre-artifact pose is clamped and the local mapper output is projected affine-free. It does not establish that the remaining local residual is learnable by the present optimizer.

## Source semantics

- The source affine state is fitted in `training/arbitrary_plane_streaming_synthetic_v7_appearance_v3.py:124`.
- V3 explicitly records the unchanged, pre-artifact pose-gauge semantics at `training/arbitrary_plane_one_shot_slide_artifacts_v3.py:86` and returns `source['state']` at line 201.
- V6 applies localized strain but returns the inherited sample/state at `training/arbitrary_plane_one_shot_slide_artifacts_v6.py:68-71`; v7 and v9 likewise replace only inputs at `training/arbitrary_plane_one_shot_slide_artifacts_v7.py:58` and `training/arbitrary_plane_one_shot_slide_artifacts_v9.py:18,53`.
- The mapper removes uniform full-canvas intercept and x/y slopes in `training/arbitrary_plane_ribbon_v6.py:13-26`.

All 512 saved generator records agree with the statement: `unchanged source affine CCF gauge fitted before v3 artifacts; not the pristine virtual-subject cutting plane and not fitted from artifact-selected pixels`.

## Reproducibility

- Frozen source/protocol commit: `460f2ba`.
- Run: `I:/AnatomyTracker/reports/v9_generator_audit_285_train_001`.
- Completion receipt SHA-256: `f11795537de7dd7ffce691e8904f69dd522966e32dfa5cdc975ffef2f8a1e293`.
- Rows SHA-256: `22e7671c22c5ef573d58110e07b348b0c09eaa4895fd0e2c0725b837fba0c6b9`.
- Summary SHA-256: `0d530fefb9463048c1d72e6ad72cf4ce2e2465b1e71434173d816fe4b0a86f3b`.
- Config SHA-256: `75831c8d6a79d7e70e3815c77bd03c669065816cc000668f2cb02b339d53cc14`.

No 284 output, checkpoint, receipt, or temporary artifact was read or changed. No DEV/CAL/FINAL animal, real data, public benchmark, or model weight was used.

## Next bounded experiment

Use one fixed eligible section to compare two explicit targets under the same mapper: (1) the current pre-artifact pose plus raw coordinate target and (2) a post-artifact affine pose plus the exactly affine-projected local residual. Overfit each without generalization claims. Failure on target 2 indicates mapper wiring/optimization trouble; success on 2 but failure on 1 isolates the gauge inconsistency measured here. A subsequent matched multi-section experiment should compare these gauges before any larger training run.
