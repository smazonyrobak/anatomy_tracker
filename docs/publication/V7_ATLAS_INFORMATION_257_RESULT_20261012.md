# 257: local full-pose atlas information survives the actual v7 artifacts

The predeclared read-only assay completed on the frozen 248 synthetic DEV panel. It did not regenerate or train on any section. For each of 227 independently drawn eligible physical sections, it compared the saved true atlas pose with 20 fixed wrong poses, using the completed 180 assay's support-matched, smoothed absolute-correlation score and an independently shuffled-atlas-intensity null. The score was given the true pose and reflection; it is **not** a blind locator.

There were 4,540 possible comparisons and 3,969 met the frozen common-support rules. Four sections had no eligible wrong-pose comparison. Equal weight to each of eight synthetic deformation plans gives:

| Changed pose factor | Eligible pairs | True pose wins | Shuffled-atlas null wins |
| --- | ---: | ---: | ---: |
| In-plane centre, ±1.5 mm | 812 | 99.15% | 52.91% |
| Through-plane offset, ±1.5/3 mm | 528 | 98.34% | 52.25% |
| In-plane roll, ±15° | 436 | 97.48% | 54.60% |
| Shear, ±0.2 | 443 | 92.61% | 55.48% |
| Raster-axis physical span, ×1.2 or ÷1.2 | 887 | 93.02% | 51.54% |
| Plane tilt, ±15° about either in-plane axis | 863 | 97.84% | 53.10% |
| **All** | **3,969** | **96.10%** | **52.78%** |

Every factor surpassed the predeclared ≥75% intact and ≥15-percentage-point null advantage thresholds, with ≥50 eligible pairs and all eight plans represented; the narrow **local-information gate passed**. Per-section intact wins were 98.7% for exact-black (28 sections with eligible pairs), 94.9% for imperfect brush (30), and 95.8% for raw/variable background (165). The lowest-visible-tissue stratum was weaker but still 84.2% across 23 eligible-pair sections. These are descriptive strata, not independent biological groups.

Together with 256's selected 4.773-mm rigid error and 2.156-mm best-of-all-128 oracle error on these same DEV sections, this isolates the present bottleneck more sharply: the v7 artifact process has **not erased nearby atlas-match information**, while the learned image-only 64-anchor head neither generates nor selects sufficiently accurate full-pose candidates. The result does not prove that globally searching all brain-intersecting planes is easy. Its true pose/reflection is privileged, candidate changes are local, all images are derived from the same pinned Allen atlas, and synthetic artifacts are not proven to match real-slide prevalence. A failed global model cannot be rescued by another scalar reranker if its near-true candidates are absent.

The next targeted architecture step is a spatially grounded, image-to-atlas correspondence representation whose fitted global plane and local residual mapping are trained together, rather than an auxiliary coordinate field attached to an unchanged anchor selector. This is supported by the prior 075 result—its auxiliary dense field improved but did not improve selected pose—and is analogous, not identical, to [scene-coordinate regression with differentiable geometric pose fitting](https://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html). [DeepSlice](https://www.nature.com/articles/s41467-023-41645-4) directly predicts image-corner atlas vectors for coronal sections; its evidence does not cover arbitrary single planes or solve local tearing. Any next model must still be assessed on independent full-sphere synthetic planes before animal-separated acquired data and later probability calibration.

The assay exited successfully. Independent read-back confirmed 227 unique physical-section IDs, eight plans, exactly 20 named wrong poses per section, 3,969 eligible pairs, four zero-comparison sections, the per-plan factor means, all output hashes, and exact frozen panel/protocol/source bindings. Its frozen output is `I:/AnatomyTracker/runs/v7_atlas_information_257_dev_001`. No CAL/FINAL biological animal, trained weights, calibrated uncertainty or public benchmark was used.
