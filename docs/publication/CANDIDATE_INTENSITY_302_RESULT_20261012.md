# Candidate 302 — hard-beam atlas intensity is usable under an oracle tissue mask

The frozen [protocol](CANDIDATE_INTENSITY_302_PROTOCOL_20261012.md) replayed the 49 already defined support-matched near/far pairs in the 299 TRAIN-inner panel. The replay verified each physical-section ID and bitwise image, then used the same intact source-tissue blocks for both candidate planes. An independent read rechecked source/protocol/panel/readout/output SHA-256 hashes, 49 distinct TRAIN IDs, all pairs and aggregate statistics. The run exited successfully; `completed.json` SHA-256 is `2ae64042e010c3b0b84310c8910ace842942510647cd3c0e97e56d694bbb9745`.

| 49 hard matched pairs | Closer-plane wins |
|---|---:|
| Direct sign-invariant atlas-intensity correlation on the shared oracle tissue mask | **35/49 (71.4%)** |
| Learned 299 atlas-intensity ranker | 29/49 (59.2%) |
| Learned 299 support-only ranker | 28/49 (57.1%) |

The direct score's median absolute-correlation margin was +0.079; median absolute near/far correlations were 0.458/0.364, against 0.039 median within-mask spatial-permutation correlation. Its near-plane correlation sign matched the generator's inversion flag in all 49. Direct and learned full scores agreed on 21 wins and six misses; direct won 14 pairs the learned score missed, while the learned score won eight that direct missed. The predeclared diagnostic threshold (≥35 wins and ≥0.05 median margin) was met exactly. Six matched pairs were ≥45° oblique, and direct intensity preferred near in all six; this small, reused synthetic subset is not a steep-oblique qualification.

This resolves a narrow question: atlas anatomy is present even within difficult candidate beams, and the 299/300 learned score has not extracted it effectively. It does **not** mean a deployable direct scorer would get 71.4%: the tissue mask comes from generator truth, the pair's common atlas support is known, the panel was reused after inspecting earlier results, and no local deformation or uncertainty calibration was tested. It also does not validate synthetic realism on physical sections.

The next controlled development step is a practical source-image tissue/reliability estimate that does not depend on automatic segmentation or a smart brush, combined with sign/contrast-robust candidate comparison. Pose refinement and local deformation should be trained jointly with an explicit cost for excessive warp, so registration difficulty can feed back into plane choice. Compare that against both the frozen first choice and support-only information on *fresh* TRAIN-inner sections before scaling training. Keep independent animal-level DEV/CAL/FINAL sealed and defer DeepSlice benchmarking. No weights, local-warp model or GUI were promoted here.
