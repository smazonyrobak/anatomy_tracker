# Real + synthetic 001: failed adaptation gate

The completed whole-model continuation from image-key **A6000** to **8000** passed independent numeric/provenance checks but **failed the predeclared adaptation gate**. Do not promote this checkpoint over A6000 or extend the same training recipe on the basis of its low sampled training loss. This is internal development with weak upstream-affine references, not biological calibration, a qualified joint model, or a public benchmark.

The run applied all 2,000 FP32 AdamW updates: eight real queries plus four generated and four prepared synthetic queries per update; equal `.5/.5` separately normalized domain losses. All 256 real training images were used, with 275/276 presentations per each of 58 donors and 43–83 per image. The six development donors were excluded from training. Real pixels were unchanged; continuous affine-render anchors used the declared, unverified fixed 50 µm PSF assumption. Whole-parent weights/optimizer were continued, without encoder merging.

## Endpoint comparison

Values below are **A6000 → final8000**. Synthetic group macros average organizational synthetic groups, not biological animals; real macros give each of six donors equal weight. Capture requires one of the top K cells to satisfy **normal ≤10° AND sign-aligned normal offset ≤500 µm**. Table values are rounded; authenticated raw artifacts and audit contain full precision.

| Synthetic scope | Rows | MAP normal (°) | MAP offset (µm) | Capture@32 | Capture@128 |
|---|---:|---:|---:|---:|---:|
| All | 640 | 39.297342 → 40.566805 | 1947.657 → 2009.147 | .726563 → .720313 | .882813 → .893750 |
| Eligible | 611 | 38.632934 → 39.674774 | 1813.213 → 1846.630 | .736893 → .736554 | .890880 → .903075 |
| Eligible, brush absent | 207 | 39.927211 → 41.361760 | 1889.294 → 1938.147 | .667917 → .686250 | .860833 → .879167 |
| Eligible, brush accurate | 204 | 38.671975 → 37.414349 | 1964.134 → 1847.638 | .799583 → .790417 | .917917 → .922500 |
| Eligible, brush imperfect | 200 | 36.758076 → 40.235687 | 1574.993 → 1727.866 | .746667 → .732083 | .900417 → .910417 |
| Censored, descriptive only | 29 | 53.526215 → 58.620716 | 4297.576 → 5309.179 | .450980 → .382353 | .666667 → .637255 |

Eligible full-catalogue NLL worsened **7.347449 → 7.446346** and normal-marginal MAP angle **37.764824 → 39.335842°**. Imperfect-brush MAP normal worsened **3.477611°**, exceeding the allowed 2° regression. The other eligible overall/mode retention conditions passed; censored rows remain in coverage accounting, not localization-success evidence.

| Real donor | Rows | MAP normal (°) | MAP offset (µm) | Capture@32 | Capture@128 |
|---|---:|---:|---:|---:|---:|
| 14452 | 11 | 36.107114 → 44.698650 | 1574.176 → 928.883 | .454545 → .636364 | .636364 → .818182 |
| 15219 | 10 | 53.582492 → 45.562037 | 2794.309 → 2731.430 | .000000 → .000000 | .000000 → .000000 |
| 15336 | 11 | 37.334344 → 40.361505 | 4167.714 → 4083.584 | .000000 → .090909 | .090909 → .090909 |
| 15439 | 11 | 31.568804 → 45.240804 | 3210.846 → 1982.296 | .000000 → .090909 | .090909 → .090909 |
| 15447 | 11 | 39.370511 → 42.407139 | 4844.363 → 3951.285 | .181818 → .181818 | .363636 → .545455 |
| 15935 | 10 | 33.549444 → 39.677846 | 3501.992 → 5045.338 | .200000 → .100000 | .400000 → .100000 |
| Equal-donor macro | 64 | **38.585452 → 42.991330** | **3348.900 → 3120.469** | **.139394 → .183333** | **.263636 → .274242** |

Five of six donor orientation means worsened. Normal error increased **4.405879°**, rather than improving by at least 5°; capture@32 improved only **.043939**, short of the required .10 absolute improvement. Both real gates fail. Across individual rows, angle improved on 21, worsened on 36 and was unchanged on seven; capture@32 gained seven rows and lost four. Better candidate inclusion and offset do not establish accurate MAP orientation.

## Training trace: progress on the sampled objective, not demonstrated global fitting

Real paired NCE averaged **.271044** in the first 100 updates and **.067672** in the last 100; the very first update was already only **.562803**. Synthetic sampled NLL was **.585818 → .569191** over those windows. These are different sampled batches, not a fixed paired evaluation. Synthetic candidate count was 47–48 (mean 47.9875), with eight real anchors added for the real objective; mean real geometric exclusions were .336375 per query. This roughly 56-key denominator is not the complete 98,304-cell gallery, and its NCE is neither a calibrated posterior nor directly comparable to full-gallery NLL.

The trace also records **12/256 unique real training anchors below 64 finite-support pixels** (738/16,000 presentations, 4.6125%). Eleven had **exactly zero finite support** (663 presentations, 4.14375%): row indices `19,32,35,48,53,78,110,125,145,163,186`; row `205` had 44.672222 pixels. Their query images are nonzero, with the eleven zero-support rows' pixel standard deviations ranging .028391–.104094. This is a concrete anchor-support concern, not proof of erroneous coordinate conventions or the cause of the aggregate failure. The trace records support, not saved rendered intensity; literal blank-anchor images must be checked separately. No rows were silently removed.

No full-gallery retrieval was measured on the 256 training images, so **donor overfit has not yet been established**. A low finite-batch loss with poor development geometry is also consistent with easy sampled negatives, exact-anchor memorization, weak-reference errors or continuous-anchor/catalogue frame mismatch. The experiment changed domain composition and continued optimization together; it does not isolate these causes.

## Next decision: one training-only gallery-closure diagnostic

Prepare one fixed comparison of **A6000 and final8000 on all 256 existing training images / 58 donors**, using the same unchanged pixels and 50 µm anchor rendering. Compare own/near-equivalent exact affine-anchor retrieval against all training anchors and against each model's already-frozen complete catalogue gallery. Retain both raster representations and exclude equivalent negatives consistently. Record own-anchor ranks/margins, geometrically eligible catalogue ranks/margins and physical capture, best available finite-frame RMS including horizontal reversal, and actual rendered intensity/support statistics; report zero/low-support strata without filtering them away.

Good sampled/anchor retrieval but poor training full-gallery capture would undermine a simple donor-overfit explanation; a large finite-frame or exact-anchor-to-gallery gap would implicate the target/gallery bridge. Good full-gallery training retrieval with the already-observed development failure would instead support a donor-generalization gap. None establishes accurate biological truth from weak affines alone. This is **diagnostic only**: no optimization, new development-donor inference, model selection on those six donors, or benchmark. A separate flat script/protocol is being prepared; it was not executed for this report.

## Independent audit and exact bindings

Root confirmed training exit 0 (601.825375 seconds; not an inference-speed benchmark) before result reads and independent audit exit 0 before its receipt was read. The audit verified full score normalization and physical metrics, 2,000 additional / 8,000 cumulative updates, scheduled donor/row/negative draws, replay and identity separation, whole-parent/source/data bindings and exact frozen tensor retention. Maximum component-score → cell-log-probability discrepancy was **3.312616e-6**; independent 12D-state → affine O/U/V reconstruction error was **1.455192e-11 µm**. No descriptor re-embedding, rendering, full training-negative-mask replay or optimizer-trajectory replay was claimed.

Frozen run: `I:/AnatomyTracker/runs/joint_v6_imagekey_real_synthetic_001`; committed launch source `3d37b71`.

| Artifact | SHA-256 |
|---|---|
| `completed.json` | `a1cd0323d9456210e64b4e622ab3375524dc2f06810b6bdd43c3e0aa49c1be31` |
| Independent `audit.json` in sibling `joint_v6_imagekey_real_synthetic_001_independent_audit` | `579b3acdbb1cb7a62ea339508cbe4091637965d7f81efcc4333d02497bdee48b` |
| Whole parent A6000 checkpoint | `280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2` |
| Parent A6000 synthetic summary | `281fa539824103abed383e5a2c20401377527c53e2bfc2156cfec9d8fb211722` |
| Parent A6000 real summary | `465bc4c932b4164f14023c1983dc9a1531f151be21e72aed000e0193e7e04e0b` |
| `joint_model_step_08000.pt` | `3c689da1a615d637048f7890e0e1e1644e9976c3f9c8447991da097f2ca7186e` |
| `gallery_descriptors_step_08000.npy` | `6bcab0980f3c332f64d179f9e7a0c53a0cea76bc750ca4e191ad2fa4d1ce81c6` |
| `allen_raw/rows.npz` | `77aa36f2ec024d1974c33c79ecee3c6d14bb49691516266a2fb4538d4d35cb57` |
| `synthetic/rows.npz` | `269fb2c5173185866741102edd8c2b7b447a0d54559cade3cd843873ddce63b3` |

The separate live native ribbon control and its operative sources were not accessed or changed during this analysis.
