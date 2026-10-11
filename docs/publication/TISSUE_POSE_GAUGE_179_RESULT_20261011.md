# 179 TRAIN-only tissue-pose gauge: empty-canvas fitting is not the blind-error cause

The predeclared audit completed on 128 independently drawn, support-intersecting full-sphere sections: two per each of the 64 TRAIN deformation bases. Its frozen output is `I:/AnatomyTracker/reports/tissue_pose_gauge_179_train_001`. No DEV, final animal, real expert label, or public benchmark was used.

The generator fits its rigid reference plane to the entire pre-artifact coordinate canvas. We compared that state with a support-weighted affine fit to the same pre-artifact coordinate field restricted to atlas-supported tissue (support > 0.25). The chart's recorded horizontal reflection was included in both fits. This is a synthetic label-gauge comparison, not biological registration accuracy.

| Stratum | Sections | Median tissue-site pose gap | 75th percentile | Mean whole-canvas residual on tissue | Mean tissue-fit residual on tissue |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 128 | 0.0299 mm | 0.0381 mm | 0.0595 mm | 0.0477 mm |
| Sparsest tissue quarter | 32 | 0.0412 mm | 0.0472 mm | 0.0619 mm | 0.0416 mm |
| Nearest-cardinal angle ≥30° | 69 | 0.0303 mm | 0.0382 mm | 0.0604 mm | 0.0494 mm |

Across all sections, the mean five-frame-point separation was 0.0947 mm, mean normal separation 0.432°, and largest observed mean tissue-site gap 0.0825 mm. The predeclared material-gap rule (overall median ≥0.25 mm or 75th percentile ≥0.50 mm) **did not trigger**. Fitting empty canvas introduces some synthetic pose-label difference, especially for sparse tissue, but its observed scale is far below the multi-millimetre blind localization errors. Relabeling or enlarging the model on this rationale is not justified.

The completion receipt bound the exact source and protocol hashes. An independent read-only check found all 128 IDs unique, all 64 TRAIN bases represented twice, both repetitions present, no ineligible or non-TRAIN rows, and matching SHA-256 values for the config, rows, summary, protocol, and audit source. The source-geometry comparison is still limited by one atlas plus synthetic deformations; it does not establish arbitrary-plane performance on unseen animals.

Next: do not promote the failed 178 spatial verifier or alter pose labels for this small gauge gap. Investigate the more plausible information and appearance bottlenecks with measured acquired-versus-synthetic tissue/background/artifact distributions and actual blind localization, while preserving separate animal-level qualification and withholding the public benchmark.
