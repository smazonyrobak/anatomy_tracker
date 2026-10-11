# 182 mapped-surface mixture: joint gradient works, pose accuracy does not pass

The frozen, TRAIN-only experiment at `I:/AnatomyTracker/runs/mapped_surface_mixture_182_train_001` completed normally. Its predeclared gate **failed in both synthetic appearances** (9 of 16 conditions false in each). Do not promote any 182 checkpoint into the GUI. This tests a bounded **supervised synthetic mapped-coordinate-risk gradient** into the direct pose predictor. It does **not** implement or validate the requested feedback from blind anatomical-fitting difficulty into pose learning.

All arms continued the same randomly initialized 132-lineage checkpoint for 300 matched updates, one independently drawn synthetic TRAIN section and two weak-affine real TRAIN sections per update. `full` allowed mapped-coordinate loss to update pose, shared image features, and mapper; `detached` stopped that loss at pose/shared features but retained mapper learning; `support_only` kept the full gradient but removed atlas intensities during this continuation while retaining atlas support. The frozen inner panel had 64 unique full-sphere physical sections per appearance, eight sections from each of eight TRAIN synthetic deformation bases held out from **182 continuation gradients only**. Those bases had been seen by parent 132. These are neither independent biological animals nor physically acquired oblique sections. Only step 300 was gated; no checkpoint was chosen from the inner results.

| Appearance and arm | Top-one mapped tissue error (mm) | Direct rigid error (mm) | Mapped ≤1.5 mm | Exact-plane map error (mm) |
| --- | ---: | ---: | ---: | ---: |
| v3 parent | 2.673 | 2.677 | 34.4% | 0.160 |
| v3 full | 2.626 | 2.629 | 35.9% | 0.153 |
| v3 detached | 2.676 | 2.679 | 34.4% | 0.154 |
| v3 support-only | **2.618** | **2.624** | 35.9% | 0.153 |
| v4 parent | 2.794 | 2.816 | 35.9% | 0.167 |
| v4 full | **2.630** | **2.642** | **43.8%** | 0.163 |
| v4 detached | 2.718 | 2.731 | 40.6% | 0.163 |
| v4 support-only | 2.697 | 2.711 | 42.2% | 0.163 |

The gradient-routing probe passed: mapped loss reached pose/shared parameters in `full` and `support_only`, not in `detached`, while all three trained mapper parameters. `full` improved mapped error over `detached` by only 0.050 mm (v3) and 0.088 mm (v4), below the frozen 0.25-mm requirement. More importantly, the matched support-only arm was slightly better than full on v3 and only 0.066 mm worse on v4. Relative to support-only, full's direct rigid error was 0.005 mm **worse** on v3 and 0.069 mm better on v4, below the 0.15-mm required gain. Zeroing atlas intensity at full inference increased its mapped error by only 0.015/0.017 mm. The matched training control is the relevant comparison; the zero-intensity inference perturbation alone is out of distribution, and support-only continuation cannot erase atlas knowledge already inherited from 132.

The supplied-correct-plane mapper remains accurate (0.153/0.163 mm), whereas blind top-one pose and mapped tissue errors remain about 2.6 mm. The blind 16 contains a ≤1.5-mm rigid candidate on 92.2% of v3 and 85.9% of v4 sections, so selection/representation and training-data realism remain material bottlenecks. These numbers do not support site-level brain-region assignment, calibrated probabilities, a GUI replacement, or a public benchmark. Additional exposure to the same synthetic atlas without better biological/coordinate evidence is not justified by 182.

An independent read-only verifier rehashed all 13 frozen outputs and the bound committed sources, checked parent/identity provenance and split rules, reproduced all 512 inner rows' aggregate metrics and every gate, and confirmed the gradient route and frozen modules. It found no material integrity or gradient-routing fault. The output receipt SHA-256 is `f790e0007dbac51b023de54e75391c388863c1084d10b0437baf6998fd179e8c`; the summary SHA-256 is `c0dcd8624d49cbb976ba90775888d0df86b23927bbb15aed627e2635696e1619`. No DEV, final-animal, calibration, or public-benchmark data were used by 182. The next priority is QC-gated biological specimen diversity and independently defensible spatial labels, without treating gross registration overlap as local anatomical truth.
