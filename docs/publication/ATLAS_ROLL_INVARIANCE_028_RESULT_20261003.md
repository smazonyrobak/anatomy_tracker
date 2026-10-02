# Independent-roll augmentation 028 — frozen result

The independent verifier passed for 3,000 scratch-training batches, 6,000 accepted fresh TRAIN sections, 192,000 positive pairs, all four checkpoints, disjoint synthetic DEV identities and exact source/checkpoint/result hashes. No 025 weights or pseudolabels were used; the only planned architecture/training change was independently rotating each atlas key patch over the full circle. The fixed DEV panel had 185 sections from eight synthetic deformation identities. Raw artifacts are in `I:/AnatomyTracker/runs/atlas_roll_invariance_028` and `I:/AnatomyTracker/runs/atlas_roll_invariance_028_development_eval`.

| Model/checkpoint | Oracle-plane top-1 within 0.5 mm | Oracle-plane top-16 | Known-point roll ±30° top-1 | Known-point tilt ±10° top-1 |
| --- | ---: | ---: | ---: | ---: |
| Unaugmented 025, batch 3,000 | **96.22%** | 99.91% | **~28%** | **~68%** |
| Roll-augmented 028, batch 1,000 | 22.81% | 66.62% | 21.19% | 22.74% |
| Roll-augmented 028, batch 2,000 | **26.75%** | 73.78% | 20.93% | 24.72% |
| Roll-augmented 028, batch 3,000 | 20.17% | 78.28% | 21.75% | 24.45% |

At the best top-1 checkpoint, raw-background recall was only **8.14%**, exact-black 54.95%, imperfect-brush 14.49%. Full-circle independent rotation made the task substantially harder for this ordinary CNN and even biased it toward the black-background mode. It did not expand roll tolerance. The preregistered ≥80% true-plane top-1 plus ≥20-point roll gain and ≤10-point tilt loss gate **failed**. Keep 025; do not merely extend 028 training or use it in a global bank.

This result does not refute proper rotation-equivariant features or explicit orientation search. It only rejects this particular invariance-by-independent-augmentation recipe at this data exposure and architecture. The separately preregistered 029 frozen group-pooling diagnostic tests whether 025's strong aligned features can be made roll-invariant without retraining. None of these component outcomes establishes unknown-plane pose, real histology performance, deformation accuracy, calibrated electrode probabilities or GUI deployment.
