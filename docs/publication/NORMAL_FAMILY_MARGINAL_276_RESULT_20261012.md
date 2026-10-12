# 276: grouping existing pose guesses by plane family helps a weak sagittal check

The zero-training diagnostic used the frozen 275 step-8,000 checkpoint. It made no new candidate, changed no network weight, and fit no temperature or orientation prior. The original decision is the highest-scoring mode/reflection branch. The new one sums the model's existing mode mass within each antipodal AP-, DV-, or ML-dominant normal family, selects the highest-mass family, then takes its highest original branch. The values are **uncalibrated decision scores**, not probabilities.

| Same frozen cases | Original first choice | Family-grouped first choice |
| --- | ---: | ---: |
| New synthetic panel, mean atlas-site error | 2.329 mm | 2.358 mm |
| Frozen 270 synthetic panel, mean atlas-site error | 2.422 mm | 2.416 mm |
| Acquired coronal TRAIN donors, weak-affine AP family | 25/28 | 25/28 |
| Acquired sagittal TRAIN donors, weak-affine ML family | 9/22 | 17/22 |
| Acquired sagittal median normal angle to weak affine | 50.98° | 38.55° |

All four predeclared 276 gate conditions passed. Eight sagittal donors changed from wrong to correct family under the weak Allen affine, none changed in the reverse direction. An independent verifier checked frozen source/protocol/parent/transfer/output hashes, 356 paired result rows and recomputed summaries, and replayed 16 synthetic decisions through the checkpoint. A second read-only audit found no pairing, arithmetic, hash, or direct affine-input leakage error. The input was the acquired image with four zero auxiliary channels; the affine defined only the readout.

This is a narrow **ranking clue**, not successful real alignment. Every acquired sagittal weak label is ML and every acquired coronal weak label is AP; the category count measures agreement with that source geometry, not position accuracy. After grouping, none of 22 sagittal normals was within 15° of its weak affine and only three were within 30°. Five of the 17 family-correct sagittal cases still exceeded 40° angular error. Synthetic first-choice site error did not materially improve, and 275's absolute capture/real-transfer gate remains failed. The same TRAIN donor set informed prior development diagnostics and the 276 hypothesis. No 276 result is an independent expert-labeled animal, physical oblique validation, calibrated electrode-region probability, or GUI-ready model.

The frozen rule is now fixed for a complementary TRAIN-donor readout under the committed 277 protocol. A positive replicate would justify further investigation of coarse plane-family scoring, but would still not establish fine pose, local warp, uncertainty, or safe GUI replacement. Do not scale unchanged pose-only training or run DeepSlice from this result.

Frozen output: `I:/AnatomyTracker/runs/normal_family_marginal_276_train_inner_001`. Protocol: `docs/publication/NORMAL_FAMILY_MARGINAL_276_PROTOCOL_20261012.md`.
