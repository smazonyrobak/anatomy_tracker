# 325: C4 loses sagittal orientation on acquired mouse sections

The frozen C4 model was shown one lower-median real section from each of 58 coronal and 40 sagittal TRAIN donors, using the prepared grayscale image and four zero auxiliary channels. These are the same donor selection and input preparation used in the earlier 272 readout. Donor, specimen, experiment and section IDs and image hashes were retained. The Allen atlas affines were read **after** each prediction: their angles and positions are approximate registration references, not independently measured physical cutting poses. The physically known coronal-versus-sagittal family is a firmer check than their exact affine coordinates.

| Acquired sections | Correct first-choice family | Weak family in top 16 | Median unsigned angle to weak affine | Median observed-midpoint gap to weak affine |
| --- | ---: | ---: | ---: | ---: |
| Coronal (58 donors) | 57/58 | 58/58 | 15.4° | 1.46 mm |
| Sagittal (40 donors) | **3/40** | **11/40** | **63.6°** | 2.79 mm |

All 40 sagittal cases had an ML-normal candidate somewhere in the full 512-branch table, but 31 first choices were DV-family and six were AP-family; only three were ML-family. This is not proof that any one candidate had the correct exact sagittal pose. It does show that candidate *scoring and top-16 retention* fail on this real orientation/domain. The [predeclared development guard](C4_ACQUIRED_FAMILY_325_PROTOCOL_20261015.md) failed four of six clauses, including the sagittal first-choice, angle and top-16 clauses.

For perspective on the same donor inputs, the frozen 270 model chose the sagittal family 19/40 with a 47.1° median weak angle ([272](ACQUIRED_TRAIN_TRANSFER_272_RESULT_20261012.md)); a separate 305 checkpoint reached 30/40 and 32.1° ([305](JOINT_SPATIAL_FEEDBACK_305_JOINT_RESULT_20261014.md)). These are different model/training lineages, not a controlled architecture comparison, but C4's 3/40 is a clear regression. C4's coronal-family count is high while its 15.4° median weak angle still narrowly misses the 15° guard. Neither cardinal-family success nor weak-affine agreement establishes arbitrary physical oblique accuracy.

**Decision:** do not scale the unchanged C4 synthetic/Lupori-only regimen. Its real sagittal deficit must be addressed with balanced acquired TRAIN-domain exposure and an animal-separated acquired guard, while preserving independently sampled full-sphere synthetic cuts and the sparse/steep failure checks. A short from-scratch intervention may proceed only with early gates for both acquired orientation families and synthetic first-choice/tail nonregression. Weak Allen affines require tolerant supervision; they must not be treated as exact physical pose truth. No current checkpoint is promoted, calibrated, or ready for the GUI or DeepSlice comparison.

Frozen result: `I:/AnatomyTracker/runs/c4_acquired_family_325_001`. Its 98 donor rows and config/row/summary SHA-256 receipts matched in a post-run audit. No DEV/CAL/FINAL donor or public benchmark was used.
