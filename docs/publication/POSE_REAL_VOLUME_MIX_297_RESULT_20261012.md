# 297 matched real-volume pose-mix pilot: failed gate

The bounded 500-batch TRAIN-inner comparison completed and the independent verifier passed. Both arms resumed the same randomly initialized-project-lineage 275 checkpoint and received the same 2,000 independently sampled v9 synthetic presentations and optimizer settings. The treatment alone received 500 additional full-angle virtual cuts from 53 TRAIN donor stacks with weak Allen-affine pose supervision at weight 0.10. Eight disjoint TRAIN donors supplied 24 fixed virtual-oblique weak-reference cuts. The fixed readouts also contained 64 new synthetic cuts, two older 64-cut synthetic panels, and 28 coronal plus 22 sagittal acquired TRAIN sections. No DEV, calibration, final, public benchmark, or pretrained external model data were used.

| Batch-500 readout | Synthetic-only control | Real-volume mix | Treatment effect |
| --- | ---: | ---: | ---: |
| New synthetic first-choice mean mapped-site error | 2.578 mm | 2.618 mm | **0.040 mm worse** |
| New synthetic top-16 capture within 1.5 mm | 54/64 | 52/64 | 2 fewer |
| Frozen 270 top-16 capture | 59/64 | 55/64 | **4 fewer** |
| Frozen 268 top-16 capture | 59/64 | 59/64 | unchanged |
| Held virtual-oblique weak-affine first-choice error | 4.542 mm | 3.956 mm | 0.586 mm better |
| Acquired sagittal weak-affine normal-family hits | 8/22 | 12/22 | 4 more |
| Acquired coronal weak-affine normal-family hits | 27/28 | 27/28 | unchanged |

Only 2/8 synthetic base means improved, whereas all 8/8 virtual readout donor means improved. On the 35 new synthetic cuts at least 30° from a principal axis, mean first-choice error was 2.49 mm for control and 2.53 mm for treatment. At batch 0 the arms gave identical per-case readouts; the divergence is due to the treatment, not different starting weights or readout sections. The predeclared gate failed on synthetic first-choice accuracy, the 6/8 synthetic-base requirement, and frozen-270 capture. This mixture is not promoted or extended merely because its weak-reference virtual metric improved.

This result is narrower than real oblique validation. The extra images are oblique *reslices* of coronal serial acquisitions approximately 100 µm apart, not physical oblique histology; their CCF coordinates inherit weak Allen section affines. Improved agreement to those weak labels and a coarse sagittal orientation-family count does not prove more accurate anatomical alignment. The absolute 3.956-mm weak-reference error remains far from usable localisation, and the synthetic top-choice tradeoff is unfavorable. The observed result supports a source-domain effect, not a high-quality arbitrary-plane model. It provides no evidence of calibrated probabilities, usable local deformation, or GUI readiness.

Independent audit: `training/verify_pose_real_volume_mix_297.py` recomputed all 36 panel summaries and nine gate clauses from raw case rows, matched 2,000 accepted synthetic and 500 accepted virtual draws to their 2,211 and 502 attempts, checked step-0 equality, verified 53/8 donor separation and no acquired-readout overlap, and SHA-256-checked 253 bound input files plus every declared output and checkpoint. All checks passed. The frozen output is `I:/AnatomyTracker/runs/pose_real_volume_mix_297_train_inner_001`; its `completed.json` SHA-256 is `4b28719d5912bb89621ad7d34b74213f17e435f90a287b3c03456e2167d1a6c6`.

Next decision: do not scale the 0.10 weak-affine mix or select its checkpoint. The main unsolved signal is anatomically reliable ranking among plausible full-sphere planes without allowing a deformation fit to reward the wrong plane (as 296 showed). Before another large training run, inspect the existing failed learned-ranker protocols and the actual-vs-generated appearance/geometry gap, then freeze a targeted candidate-comparison objective with held-donor readouts and a strict synthetic regression gate. Keep physical arbitrary-oblique validation as an explicit unmet requirement.
