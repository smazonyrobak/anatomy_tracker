# 189: acquired fluorescence contains a local atlas-fit correction direction

The [predeclared TRAIN-only assay](LUPORI_ACQUIRED_ATLAS_FIT_DIRECTION_189_PROTOCOL_20261011.md) completed on 64 valid physical Lupori sections, 16 deterministically chosen from each of the four TRAIN animals. It used the frozen 188 step-1200 model, exact image-wide-p99 fluorescence input, and finite-thickness Allen 2017 atlas rendering. It made **no weight updates** and read no Lupori DEV/CAL/FINAL animal or public benchmark. The human VisuAlign planes were used only *after* the blind 16-branch proposal to choose a diagnostic near candidate and measure direction; they were never supplied to the score.

| condition | eligible | actual atlas intensity favors the move toward reference | within-support permuted atlas intensity favors it | actual atlas derivative favors it |
|---|---:|---:|---:|---:|
| highest-prior branch, when initially 0.5–3 mm from reference | 56/64 | 43/56 (76.8%) | 29/56 (51.8%) | 38/56 (67.9%) |
| retrospectively closest captured branch in that range | 61/64 | **47/61 (77.0%)** | 29/61 (47.5%) | **48/61 (78.7%)** |

All 61 diagnostic near-branch moves changed the five-site physical error monotonically in the intended direction, and the minimum fixed-mask atlas-support retention was 96.8%. The diagnostic branch's real-intensity direction win fraction exceeded 50% for **all four** TRAIN animals (13/16, 14/16, 10/14, 10/15). The predeclared screen passed. Of the 64 sections, 51 had a top-16 branch within 1.5 mm of the approximate reference.

This is evidence that **real acquired fluorescence carries a locally useful atlas-anatomy signal once a near pose is captured**. It does not show that the signal can select the near pose from a blind candidate set. It did not exercise the learned deformation head, test an anatomical-fit gradient into pose weights, or validate steep-oblique physical sections. The human planes are approximate and all four animals' sections are near coronal; source fluorescence is not the same modality as the atlas template. In particular, the observed direction percentages are not calibrated predictive probabilities.

The next bounded test must compare near and wrong blind proposals on *identical support-matched pixels*, against a within-support intensity permutation and the original pose prior. Only if candidate selection has an anatomy-sensitive advantage should fitting effort and constrained deformation be connected to pose learning in a matched feedback-versus-detached experiment. No GUI promotion or DeepSlice benchmark follows from 189 alone.

The post-exit read-only verifier matched all frozen output hashes, protocol/source/checkpoint identities, 64 source-image hashes, distinct section IDs, candidate ordering, per-branch score signs, physical monotonicity, per-animal counts and the gate. Frozen `I:/AnatomyTracker/runs/lupori_acquired_atlas_fit_direction_189_train_001/completed.json` SHA-256 is `38133530a4b1bf465f717931925fbe4ea34c42705be06373e966b4e3a66dcd4f`.
