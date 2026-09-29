# Native ribbon local 001: limited pose progress, failed local gate

The 2,000-update conditional joint pose + 3D ribbon control **failed its predeclared gate**, independently confirmed after process exit. Eligible oracle centre error improved **814.235657 → 716.556673 µm (11.9964%)**, short of 20%; slab error likewise improved only **814.237311 → 716.559497 µm**. All per-mode/per-subject mean nonregression conditions passed. Selected centre error improved 68.7645%, but this largely reflects learning which horizontal-raster branch to choose, not solving pose or individual anatomy.

This control used the whole original image-key 4000 parent, zero-initialized coordinate/ribbon heads, frozen retrieval-producing tensors, known PSF and truth-near initial poses. Eight training and four held-out **synthetic coherent subjects from one atlas** are not biological validation. All 384 development observations were retained; 270 eligible observations enter the gate and 114 censored observations do not. No global retrieval, uncertainty calibration, public benchmark or deployment claim follows.

## What improved

Eligible equal-subject macros are below; distances are visible-support-weighted Euclidean centre errors in µm. The selected geometric initializer uses a fixed identity-reflection tie, while the oracle initializer uses recorded reflection. These are not the step-zero model's own score selections.

| Eligible scope | Rows | Selected initial → final | Oracle initial → final | Plane-normal initial → final (°) |
|---|---:|---:|---:|---:|
| Overall | 270 | 2685.479 → 838.822 | 814.236 → 716.557 | 6.138344 → 6.102826 |
| Raw | 91 | 2697.586 → 876.632 | 816.780 → 722.834 | 6.144613 → 6.135283 |
| Exact black | 91 | 2697.586 → 839.621 | 816.780 → 708.983 | 6.144613 → 6.088042 |
| Imperfect brush | 88 | 2662.321 → 797.175 | 809.252 → 718.131 | 6.125254 → 6.083802 |
| Held-out subject 00000000 | 64 | 2645.936 → 876.330 | 852.411 → 706.333 | 5.826826 → 5.627871 |
| Held-out subject 00000001 | 66 | 2906.371 → 768.133 | 783.867 → 696.399 | 6.054569 → 6.002487 |
| Held-out subject 00000002 | 75 | 2484.174 → 871.099 | 842.889 → 775.020 | 7.008387 → 7.119875 |
| Held-out subject 00000003 | 65 | 2705.435 → 839.725 | 777.775 → 688.475 | 5.663596 → 5.661071 |

Subject IDs share prefix `joint-v6-coherent-cohort-002-development-subject-`. Eligible reflection accuracy rose **40.8565% → 94.8141%** against the fixed geometric tie (the step-zero model's own score accuracy was 30.6686%). Raw / exact-black / imperfect-brush endpoint accuracies were **93.3182% / 94.5909% / 96.6190%**. Remaining selection errors still raise endpoint centre error from oracle 716.557 to selected 838.822 µm. The oracle normal error improved only .035518°, and subject 00000002's normal worsened .111488°.

## Saved-coordinate attribution: mainly in-plane pose, little net field benefit

A separate **posthoc CPU calculation** reconstructs final planar coordinates from saved final states, deletes the saved field only at the endpoint, applies the recorded raster reflection and scores against exact original coordinates with identical visible weights. It is **not** an ablated recurrence rerun or a changed gate.

| Oracle endpoint construction | Centre mean (µm) | Normal component (µm) | Tangential component (µm) |
|---|---:|---:|---:|
| Geometric initializer | 814.235650 | 355.755900 | 684.815222 |
| Final pose, zero field | 718.853289 | 353.011162 | 576.526421 |
| Final pose + saved field | 716.556672 | 354.991202 | 572.147848 |

Thus **95.382 µm of the 97.679 µm mean reduction (97.65%) is present in final plane geometry**; adding the final field improves only 2.297 µm net and slightly worsens the normal component. This is genuine coordinate/pose progress, primarily in-plane, not predominantly deformation hiding a bad pose. Because the field influenced earlier renders, this arithmetic cannot attribute independent causal training contributions. In the imperfect-brush subset, adding the final field actually worsens centre error **716.532 → 718.131 µm**; on held-out subject 00000003 it worsens **684.217 → 688.475 µm**.

The oracle final-state centre shifted by **296.715 µm** on average and the complete frame rotated **.868051°**; visible planar coordinates moved **323.717 µm**. The final field moved visible points **72.310 µm** on average, but its directions produced little net error reduction. Oracle director error also failed to improve: **.113505** at step zero versus **.114881** at the endpoint. Normal/tangential components above are relative to the fitted target plane, not local curved-surface normals.

## Pose updates and field limiter

For **810 correct-branch updates** (270 eligible observations × three updates), absolute median / p95 were:

- Normal tangent parameters: `.001477 / .004766`, `.003212 / .013198`; normal-offset increment **4.272 / 10.504 µm**.
- In-plane roll: `.001830 / .008148 rad`; in-plane translations **7.326 / 27.604 µm** and **69.108 / 346.835 µm**.
- Log-span increments: `.010063 / .027897`, `.006977 / .021946`; shear `.000550 / .001693`.

No saved pose increments reached 99% of their configured caps. The response is strongly weighted toward one in-plane translation and spans, with small normal/offset updates; a large centre displacement must not be mistaken for successful out-of-plane alignment.

The derivative limiter was active on **772/810 correct-branch states (95.3086%)**, median scale **.526431** (p05 .337411, p95 .952463). The other branch had **786/810 (97.0370%)**, median .436159; pooled both-branch activity was **1558/1620 (96.1728%)**, median .473263. These denominators include three active field states per eligible observation, not the zero-field initializer. The minimum scale over all retained rows was .231249; maximum saved derivative bound was .350000024 (floating-point agreement with .35). Thus pressure is not confined to the branch lacking geometric supervision. Raw surface tanh cap fraction averaged 3.34685% over eligible subjects; director cap fraction was zero.

All recorded outputs were finite and the run's canonical topology/orientation checks passed. These observations do not justify relaxing the deformation bound: the useful field benefit is small, normal/offset learning is weak, and the accepted targets had already passed the separate representation-capacity check. Keep this as a failed conditional pilot; diagnose pose and director learning before claiming successful joint registration or simply expanding field freedom.

## Independent verification and provenance

Training exited 0 before artifact inspection; independent audit also exited 0, with integrity true and local gate false. It verified raw coordinates/mean metrics, original data and identities, schedule/update budget, parent/source bindings and exact frozen retrieval tensors. Maximum mean-metric discrepancy was **.00187565 µm**, reconstructed-coordinate discrepancy **.00581836 µm**. Weighted p95 differed by up to **2.76136 µm** at CDF ties; p95 is diagnostic, not a gate. Topology was authenticated from saved reported checks, **not independently replayed**. Evaluation elapsed time 831.847139 seconds is not an inference-speed benchmark.

Frozen run: `I:/AnatomyTracker/runs/joint_v6_ribbon_local_001`.

- Completion SHA-256: `a2bee6dc1ffcaafcef3b4ad4a49223653f582f41ab071a3ca25928e7db7a83d7`.
- Independent sibling `joint_v6_ribbon_local_001_independent_audit/audit.json`: `46b243156dc6e9388d7aa9dc2dac362f557b25a183c25226c0fefa5167b1f21d`.
- Posthoc script `I:/AnatomyTracker/runs/joint_v6_ribbon_local_001_posthoc/analyze_native_ribbon_001_frozen.py`: `f30c987967c30113df4855edfe3507b4ff51cd3ee7dd246b738de87305d3e89f`.
- Posthoc numeric result `I:/AnatomyTracker/runs/joint_v6_ribbon_local_001_posthoc/native_ribbon_001_frozen_analysis.json`: `f6d4a3b2ed4a8edfc9bbb706fcbb90b0dcd09f5c36e2c23429e7f62cd2a909ee`.

No extra model inference, training, thresholds or benchmark was introduced for this analysis.
