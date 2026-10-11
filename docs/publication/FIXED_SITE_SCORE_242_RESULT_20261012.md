# 242: fixing the support-count score loophole did not recover useful pose feedback

The frozen fixed-site ablation completed without training on the exact 32 development images used in 240. All 32 physical-section IDs were distinct; their input hashes, the 240 checkpoint/manifest/source hashes, this script, the predeclared protocol, the raw-row hashes, and independently recomputed means matched the completion receipt. The sealed fresh 64-section readout was not opened.

| Physical tissue-site error (mm) | All | v4 | v7 |
|---|---:|---:|---:|
| Near branches in blind top 16 | 17/32 | 8/16 | 9/16 |
| Near before / after intact fixed-site fit | 1.307 / 1.307 | 1.342 / 1.342 | 1.275 / 1.275 |
| Near after support-preserving atlas-intensity shuffle | 1.307 | 1.342 | 1.275 |
| Known-true before / after intact fixed-site fit | 0.187 / 0.210 | 0.180 / 0.180 | 0.194 / 0.240 |
| Known-true after shuffled-intensity fit | 0.257 | 0.180 | 0.334 |
| Near / known-true accepted intact updates | 0/17 / 3/32 | 0/8 / 0/16 | 0/9 / 3/16 |

The predeclared mechanism criteria fail: near correction is zero rather than ≥0.25 mm, intact atlas offers zero near-fit advantage over the shuffle rather than ≥0.15 mm, and known-true error is 0.210 mm rather than ≤0.20 mm. The 240 score's changing-support/top-64 behavior was consequential: freezing its initial sites removes nearly all accepted motion, while the old score accepted 16/17 near and 17/32 known-true moves, many harmful. But this is **not** a clean causal estimate of support bias alone. Requiring *every* initial site to remain supported rejected 566 known-true and 405 near optimization trials across intact arms, so the ablation also makes the feasible local search much narrower. It shows that simply closing the loophole does not yield a workable corrector.

Decision: do not use either 240's changing-support score or 242's fixed-site gradient-ascent score for the final pose-feedback loop, and do not spend more training presentations on the unchanged 240 descriptor objective. A next architecture should start by correcting the source feature/label lattice, learn **local 2-D-to-3-D correspondences** with physically close hard negatives, and estimate a plane from geometrically consistent matches with a no-shift fallback rather than optimizing an unconstrained scalar score. This direction is analogous to [SLIV-Reg's feature matching plus robust pose estimation](https://arxiv.org/abs/2410.18683), although that paper evaluates same-modality CT/MRI and does not prove transfer to histology. [Histology-to-micro-CT work](https://www.nature.com/articles/s41598-025-11583-w) likewise separates learned cross-modal similarity, plane initialization and deformation, but used an orientation prior; it is not evidence of unrestricted-angle success. A conditional mechanism must be proven before adding joint pose/deformation feedback and before any animal-level or DeepSlice claim.

Frozen output: `I:/AnatomyTracker/runs/learned_3d_fixed_site_242_validation_001`. SHA-256: `completed.json` `64419ad98347c86dc726cb91cfc1c7992d917be1ab5dda7de6773799e24250f7`; the completion receipt binds the raw rows, summary, script, protocol, manifest and checkpoint. No public benchmark, reserved animal, probability calibration, or GUI deployment occurred.
