# 327: balanced real exposure repairs cardinal-family transfer but harms sparse cuts

The new random C4 model completed exactly 1,000 optimizer batches in 9.22 training minutes. It saw 4,000 accepted independent v10 TRAIN synthetic cuts (4,122 attempts) and 1,000 weak acquired presentations, balanced as 334 Lupori, 333 Allen coronal and 333 Allen sagittal. No earlier weights, features or pseudolabels were loaded. The acquired TRAIN-inner readout used 10 coronal and eight sagittal donors excluded from acquired gradients; the synthetic readout was the *same frozen panel* used by 321. These are development checks, not physical pose ground truth or a public benchmark.

| Batch-1,000 readout | 321 same panel, batch 1,000 | 327 batch 1,000 | 327 gate |
| --- | ---: | ---: | --- |
| Inner first-choice mapped mean (128) | 3.720 mm | 3.500 mm | Pass |
| Inner cases over 5 mm | 19 | 14 | Pass |
| Steep-oblique mapped mean (32) | 3.804 mm | 3.901 mm | Pass, within 0.20 mm |
| Sparse/low-visibility mapped mean (32) | 5.556 mm | **6.833 mm** | **Fail**, >0.20-mm allowance |

On the held-out acquired TRAIN donors, 327 chose the known AP family for **10/10 coronal** and the ML family for **8/8 sagittal** sections; both families were present in the top 16 for every section. Their median unsigned normal gaps to the *weak Allen affines* were 7.08° and 3.08°. This is a useful domain-transfer signal, not evidence of exact physical plane recovery: the weak affines are approximate and the readout contains only cardinal acquired orientations. The 321 batch-12,000 guard had selected sagittal family for only 3/40 different TRAIN donors, so the two observations are not a paired final-model comparison.

The [predeclared 327 gate](C4_BALANCED_TRANSFER_327_PROTOCOL_20261015.md) failed exactly one of eight clauses, the sparse mean. The learned local map still barely contributes: mapped-minus-rigid mean error was approximately 0.00000 mm inner, +0.00163 mm steep, and −0.00395 mm sparse. Its true-state robust map loss was flat at 0.0298 to 0.0297 mm from the first to last 100 batches; that robust loss is **not** an ordinary Euclidean coordinate error. The current model is neither a useful joint-deformation solution nor a deployable arbitrary-plane estimator.

The post-exit audit matched every output hash in `completed.json`, source/protocol/model and frozen-panel hashes, 1,000 consecutive training rows, 0/1,000 readouts, all 210 final case rows, 4,000 accepted synthetic draws from TRAIN bases 0–55, and disjoint acquired-gradient/inner donor IDs. Raw rows independently reproduced the three synthetic means and two acquired family counts. Frozen directory: `I:/AnatomyTracker/runs/c4_joint_balanced_327_001`; completion SHA-256 `a6eedd820a11cb0e455508b0f069f3a73cea51ec46545bf85c1b5b55acc34a87`. `transfer_feasibility_gate_passed=false`, `calibrated=false`, `model_promoted=false`.

**Next efficient test:** keep this exact random initialization, synthetic draws, donor split and objective, and reduce *only* the acquired weak-loss weight from 0.3 to 0.1 in a bounded matched arm. This tests whether the cardinal-family gain and sparse-cut regression trade off with real-supervision strength. The cross-arm reuse is a controlled comparison of separate models, not repeated augmentations within either training run. Do not extend 327, scale the model, calibrate it, or run DeepSlice from this result.
