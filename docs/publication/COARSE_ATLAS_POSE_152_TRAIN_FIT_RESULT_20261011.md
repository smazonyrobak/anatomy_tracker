# 152 TRAIN-only pose-fit ridge check

On 128 newly drawn, independently artifacted synthetic TRAIN sections from four withheld TRAIN deformation plans, the [flat diagnostic](../../training/diagnose_train_pose_fit_ridge_152.py) fitted **known true rigid correspondences** at valid 24×24 image sites. Every section was evaluated under both raster-reflection representations and four prespecified ridge penalties. All 256 design matrices had full rank. The 200 draw attempts, 128 accepted sections, source-plan identities, output hashes and row summaries passed an independent read-only audit.

| Plan-equal mean native-256 rigid error, mm | Fixed 150 ridge (24,12,12) | Center-only | Slope-only | No ridge |
| --- | ---: | ---: | ---: | ---: |
| Correct reflection; candidate selected at 0.35–1.5 mm | 0.301 | 0.198 | 0.131 | <0.000001 |
| Opposite reflection of the same candidate | 2.968 | 0.198 | 2.808 | <0.000001 |

The fixed ridge strongly pulls the fitted plane toward the starting pose even when the supplied rigid correspondences are exact. Repacking the fitted affine plane into the internal frame adds no measurable error. The opposite-reflection candidates started at **3.849 mm mean** error (range 1.288–7.524 mm); only 2/128 were initially within 0.35–1.5 mm, so the second row is not a matched nearby-candidate comparison. With full-rank, noiseless rigid targets, the unregularized solution is algebraically expected to recover the plane. **This does not justify removing regularization for noisy learned matches.** The next model should train and assess a less biased, confidence/conditioning-aware fit on fresh TRAIN geometry, and compare on a new DEV panel.

The output is `I:/AnatomyTracker/runs/coarse_atlas_pose_152_train_fit_ridge`; raw rows SHA-256 is `4d8622f020e181a64c248fe2608337387adb5016837e6dd98f42bf9922d59355`. The receipt binds the runner, sampler, artifact v3/v4 and 150 matcher source, but does not hash every transitive geometry/loader import; the committed source revision supplies that broader code snapshot. No learned model, DEV case, expert truth, final animal or public benchmark participated in this diagnostic.
