# 058 normal-anchored joint model: failed development gate

The randomly initialized lineage continued 019 batch 18,000 for 12,000 batches with a new 64-anchor antipodal plane-normal head and the existing finite-thickness atlas-conditioned mapper. The completed run used 24,000 independently drawn eligible synthetic TRAIN planes from 64 synthetic deformation bases and 12,000 distinct real TRAIN sections from 1,885 donors. Of those real sections, 4,564 had never appeared in an earlier run; 7,436 had, but no real section was repeated within 058. Animal/specimen/experiment/section identifiers and source bindings were retained. There was no public-benchmark, calibration or final-test use.

The fixed 185-section/eight-synthetic-identity and 64-section/six-real-donor development evaluation failed every pose-capture and real-retention condition. Values below weight identities equally; real values are disagreement with inherited weak Allen affines, **not expert anatomical error**.

| Checkpoint | Selected synthetic tissue-to-CCF error | Best-eight rigid oracle | Exact-pose mapped error | Weak-real five-point disagreement |
| ---: | ---: | ---: | ---: | ---: |
| 0, exact 019 reproduction | **2.591 mm** | **1.175 mm** | 0.129 mm | **0.595 mm** |
| 2,000 | 3.015 mm | 1.274 mm | 0.129 mm | 4.742 mm |
| 6,000 | 2.685 mm | 1.300 mm | 0.130 mm | 4.580 mm |
| 12,000 | 2.809 mm | 1.252 mm | 0.129 mm | 4.506 mm |

A separate frozen, truth-scored candidate readout distinguishes the failure mechanisms. Across **all 64 new normal-anchor modes and both reflections**, the physically best synthetic candidate improved from 3.062 to 1.671 mm but remained worse than the old 16-mode head's 1.027→1.062 mm. Among only the new head's eight highest-prior branches, best synthetic error improved from 3.711 to 2.458 mm—still not competitive. On real weak references, even the physically best new candidate remained 4.349 mm at batch 12,000, versus 0.798 mm for the best old candidate. The old head also regressed modestly: its selected weak-real disagreement rose from 0.584 to 1.006 mm. More importantly, the combined image prior selected a new anchor branch on **all 64 real DEV images** at every trained checkpoint, despite those branches being grossly misplaced. New anchors rarely won the synthetic top-one prior (about 2–3% at trained checkpoints).

The new head's mass supervision rewarded closeness of the **plane normal only**, not the full spatial plane. For near-coronal real images it could therefore assign high mass to a normal-compatible branch with a wrong position and scale. This is a concrete training-objective flaw, not just an uncertain visual impression. Training-only near-true-normal candidates made the batch-level fit error optimistic and did not transfer to truth-blind top-eight inference. The atlas-fit-to-anchor gradient was nonzero on the first batch, and exact-pose local mapping remained accurate, but neither fact establishes global pose learning.

Both runners exited successfully. Post-exit checks matched the training config/log/draw hashes and all four checkpoint hashes in both evaluation receipts. The 996 raw rows in each evaluation reproduced every identity-equal summary cell. All 24,000 accepted synthetic physical IDs and 12,000 real section IDs were unique within the run; real TRAIN and DEV donor IDs were disjoint. The single PyTorch tensor-to-scalar warning concerned logging, not the optimizer.

Decision: do not deploy or continue this objective unchanged. Keep 019 batch 18,000 as the internal reference. The next substantive continuation must preserve the existing old-head performance while training new candidates, supervise probability mass by **full pose quality** (not normal alone), and give the new head enough real and arbitrary-plane synthetic exposure to learn position. Inspect candidate capture and real retention before any calibration or GUI promotion. No DeepSlice or untouched final-test labels were consulted.
