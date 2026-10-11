# 243: the frozen learned descriptors do not yield reliable 3-D correspondence geometry

The precommitted local-match/robust-consensus check completed on the exact 32 TRAIN-inner development sections from 240, with no training updates. The independent post-exit audit matched all 32 distinct physical-section IDs, raw/summary hashes and recomputed means, plus the checkpoint, manifest, protocol, matcher and fitter hashes. The known-true role is a diagnostic, and the near role is selected by truth only *after* the blind top-16 proposals exist; neither is a blind end-to-end prediction. The 240 fresh 64-case panel remains sealed.

| Physical tissue-site error (mm) | All | v4 | v7 |
|---|---:|---:|---:|
| Near branches in blind top 16 | 17/32 | 8/16 | 9/16 |
| Near before / after intact learned-match consensus | 1.307 / 1.314 | 1.342 / 1.451 | 1.275 / 1.191 |
| Near after support-preserving intensity-shuffle consensus | 1.294 | 1.374 | 1.223 |
| Known-true before / after intact consensus | 0.187 / 0.263 | 0.180 / 0.264 | 0.194 / 0.262 |
| Near intact accepted fits | 5/17 | 2/8 | 3/9 |
| Near valid-site correspondence error, intact | 1.652 | 1.611 | 1.678 |

The predeclared criteria fail: intact mean near error **worsened by 0.007 mm**, the null fit was slightly better than intact rather than ≥0.15 mm worse, and the known-true pose drifted above the ≤0.20-mm requirement. Robust three-point consensus cannot recover a good plane from these matches. On the near roles, the local search found some atlas support for a mean 237 of 256 query sites, but only 62 sites per case had surviving synthetic tissue truth; the score has no learned source-side matchability gate. More importantly, even on those valid labelled sites the mean match error was 1.652 mm, larger than the initial near-plane mean 1.307 mm. Mean selected cosine was negative (−0.347), and the intact/null raw cosine means were very close. These numbers establish a weak descriptor, not merely an over-eager fitter. The label-coordinate offset identified in 240 was intentionally preserved here to test those exact frozen weights; correcting it requires new training, not post-hoc relabeling of this result.

Decision: do not extend or deploy the 240 descriptor or either 242/243 corrector. The next model needs a **fresh random initialization** with a correctly centred source/atlas coordinate lattice, source-side tissue/reliability weighting that works without a mandatory smart-brush mask, local 3-D hard negatives at the submillimetre-to-few-millimetre scales relevant to the goal, and a geometric pose/deformation objective evaluated in physical millimetres. Candidate fit must inform pose without allowing flexible warp to explain a wrong plane. This is a targeted architectural change, not another exposure or threshold tweak. It must later be trained into one standalone checkpoint, then tested on independent animals; the present result licenses no GUI replacement, numerical electrode-region probability claim, arbitrary physical-oblique accuracy claim or DeepSlice comparison.

Frozen output: `I:/AnatomyTracker/runs/local_3d_learned_consensus_243_validation_001`. SHA-256 `completed.json`: `3469c0a621289d8c416eb52f2c364ab16c9d40df86f49fd08d9a8fd5a4c62d04`. Its receipt binds the raw rows, summary, source, protocol, validation manifest and checkpoints. No reserved animal, public benchmark or calibrated probability was used.
