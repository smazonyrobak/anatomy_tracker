# 053 result: the previous local control used oracle-warped atlas geometry

The frozen 025 descriptor was evaluated twice on each of the same 185 eligible synthetic DEV sections (eight held-out synthetic deformation identities). Query pixels, image patches, 256 regular plus 32 exact-pixel atlas candidate locations, known full-frame plane, and section-specific finite-thickness PSF were identical. Only the atlas render changed: (A) the exact synthetic per-pixel tissue-to-CCF surface used in the original 025 control, versus (B) the *correct rigid plane* from the true full-frame state and reflection. Neither surface was predicted by the model in this diagnostic. No image was opened or inspected.

| Identity-equal local retrieval at 0.5 mm | Oracle-warped atlas | Correct rigid atlas |
| --- | ---: | ---: |
| Top-one point recall | 96.22% | 75.20% |
| Top-16 point recall | 99.91% | 97.42% |
| Exact-pixel key ranked first | 91.99% | 59.67% |
| Mean exact-pixel image/atlas descriptor cosine | 0.897 | 0.742 |

The predeclared 20-percentage-point loss gate is met: rigid top-one falls by **21.01 points**. The warped-to-rigid coordinate difference averages 141 µm on identity-equal queries and has query-level median 100 µm, 90th percentile 302 µm, and 99th percentile 548 µm. Pooled rigid top-one recall falls with this difference: 89.1% at <100 µm, 67.7% at 100–250 µm, 52.4% at 250–500 µm and 22.7% at ≥500 µm. This trend is descriptive, not a matched causal manipulation of deformation severity; the paired A/B render is the causal geometry control.

The 025/052 positives were therefore easier than the rigid atlas search that would actually be available. The 96% oracle-warped local result should **not** be described as true-plane rigid retrieval capacity. A rigid true-plane local top-16 recall near 97% shows that recoverable local evidence still exists when the plane is supplied. It does not reconcile the 030/052 whole-brain top-16 recall of only 12–15%, where the bank additionally quantizes 3D location, angle, scale and PSF and introduces roughly two million lookalikes. An anatomical point may be locally recoverable yet lose the global ranking competition.

The evaluator exited normally; SHA-256 read-back matched its frozen config, rows and summary. All 370 expected result rows are present (185 paired sections × two arms), with eight distinct synthetic DEV identities. It reused the same frozen 025 checkpoint and DEV panel; no training, final-test animal, probability calibration, public benchmark or GUI deployment occurred. The query pixels are selected using the synthetic true tissue mask and are not a deployable input-selection method.

Next targeted change: train cross-modal descriptors with rigid atlas positives at known anatomical positions while the *observed image alone* carries synthetic warps, and supervise nearby physical matches tolerantly. Then separately measure the impact of bank orientation/scale/PSF discretization before rebuilding a costly whole-brain bank. Continue judging progress by unknown-plane capture and selected tissue-coordinate error, not local oracle recall alone.

Frozen output: `I:/AnatomyTracker/runs/rigid_vs_warped_atlas_patch_053_development_eval`.
