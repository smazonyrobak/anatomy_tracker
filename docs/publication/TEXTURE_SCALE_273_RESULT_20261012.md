# 273: fixed low-pass texture comparison — negative result

The frozen 270 model saw three fixed versions of each of the same 98 acquired TRAIN sections and 41 no-brush fresh synthetic TRAIN-inner sections. Its raw acquired output exactly reproduced 272's casewise normal-family and angle readouts. A post-exit audit matched all three output hashes and counted 417 unique section/arm rows (139 per arm). No image, checkpoint, or label was changed between arms except a fixed 7×7 grayscale Gaussian blur; the four auxiliary channels remained unchanged.

| Input | Coronal family correct /58 | Sagittal family correct /40 | Sagittal median normal gap to weak affine | Synthetic top-16 ≤1.5 mm /41 | Synthetic mean first-choice error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 56 | 19 | 47.1° | 28 | 3.116 mm |
| Gaussian σ=1 pixel | 58 | 20 | 48.7° | 27 | 3.205 mm |
| Gaussian σ=2 pixels | 58 | **9** | **70.6°** | **8** | **4.368 mm** |

Neither blur meets the preregistered ≥10/40 sagittal gain, ≤3/58 coronal loss, and ≤4/41 synthetic capture loss. Heavy blur is actively harmful. Removing fine texture with a simple low-pass filter is **not** the explanation or a justified inference-time fix for the real sagittal failure. The weak Allen normal is only an approximate reference, and this TRAIN-only comparison is not biological validation. Do not search additional unplanned filters on these same cases; investigate a measured representation/appearance mismatch and retain raw images for local anatomy and electrode mapping. The joint model, calibrated uncertainty, and GUI replacement remain untrained/unshipped.

Frozen output: `I:/AnatomyTracker/runs/texture_scale_273_train_001`. Protocol: `docs/publication/TEXTURE_SCALE_273_PROTOCOL_20261012.md`. No acquired DEV/CAL/FINAL labels or public benchmark were used.
