# 048 candidate-evidence ranker: negative development result

The frozen scratch-lineage 019/041/046 model generated and mapped eight proposed planes per image; a newly initialized 11→32→1 head learned only to select among them. Training completed in 607.7 s on 1,000 accepted fresh arbitrary-plane synthetic TRAIN sections (1,106 independently recorded draws) from 64 synthetic animal/deformation identities. Peak GPU allocation was 1.35 GB. No paired background variants, outside model weights, real or public benchmark labels, final-test animals, or uncertainty calibration entered the stage.

The frozen three-checkpoint comparison completed in 97.3 s on all 177 eligible sections of the unchanged, disjoint eight-identity synthetic 037 DEV panel. Errors are observed-tissue-to-CCF distances, equally averaged over identities:

| Selector | Selected mapped error |
| --- | ---: |
| 019 prior top one / 048 batch zero | **2.450 mm** |
| Previous fitted atlas score | 2.466 mm |
| 048 batch 500 | 2.451 mm |
| 048 batch 1,000 | 2.487 mm |
| Physical best of eight (unavailable oracle diagnostic) | 0.943 mm |

Neither trained checkpoint approaches the predeclared ≥20% improvement or ≤1.5-mm absolute gate. At batch 1,000 the raw, exact-black, and imperfect-brush selected errors are 2.362, 2.613, and 2.404 mm, versus prior 2.274, 2.780, and 2.235 mm. The trained scalar evidence head is **not promoted**. The best-eight branch remains much better than what the model can identify, but even that best branch is far from the ~0.13-mm known-pose mapping control documented for 019.

This is the third distinct failed attempt to choose widely separated predicted planes from local atlas-fit evidence (020, 021, now 048); 047 also showed that adding a differentiable fit loop does not rescue this selection. The narrower inference is that the tested eleven candidate-level summaries and current matcher do not reliably discriminate correct anatomy over a roughly millimetre-wide proposal gap. It is not proof that a single histology image lacks information. Do not spend another continuation on a scalar head over these same candidate planes. Prioritize **capturing a near-correct arbitrary plane** with a stronger global image/atlas representation and explicitly test near-truth proposal recall before any finer deformation/ranking stage. Use a fresh synthetic identity holdout if that capture improves; expert animal-level truth remains essential.

A read-only audit confirmed 1,000 accepted presentations, 1,106 unique physical draw IDs, disjoint 64 TRAIN and eight DEV synthetic animal IDs, 177 finite DEV rows, and matching draw/row SHA-256 receipts. Frozen artifacts are `I:/AnatomyTracker/runs/candidate_evidence_048_pilot` and `I:/AnatomyTracker/runs/candidate_evidence_048_development_eval`. Draw hash `3134df011c88dd3683f28e1e40285b9fb59a84fa60b53b1458061ebaad0f67c0` and row hash `884e9978ef8475a93909759d00e1b639fe306f03e16cf22c1da5d31894d3f1b5`.
