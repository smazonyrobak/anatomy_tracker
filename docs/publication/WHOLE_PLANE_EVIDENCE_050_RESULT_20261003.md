# 050 whole-plane evidence: true plane recognizable, proposed planes not selectable

The frozen scratch-lineage 050 run trained for 5,000 batches on 5,000 independently sampled synthetic TRAIN physical sections (5,496 total draws, including 496 eligibility rejections). All 5,000 accepted physical IDs are distinct, use 64 TRAIN deformation bases, and have no animal-ID overlap with the eight synthetic DEV identities. The model continued the randomly initialized 025 patch encoder and trained a new spatial whole-plane comparison head; the 019 pose generator remained frozen. Training finished normally in 573 seconds at 1.38 GB peak GPU memory. Checkpoint, source, training-log, draw-log, DEV-row and DEV-summary hashes matched their frozen records. No image was opened, no final-test/public labels were used, and normalized scores are not calibrated probabilities.

On the same 177 eligible arbitrary-plane synthetic DEV sections (eight identity-equal deformation plans), the 019 top-eight rigid candidate oracle remains 1.001 mm. The new head selects only a negligible improvement over the 019 image prior, despite recognizing the **injected exact synthetic plane** almost perfectly:

| 050 batch | Selected rigid tissue error | Exact plane beats support-matched ≥2-mm wrong plane | Weak-real selected five-point disagreement |
| ---: | ---: | ---: | ---: |
| 0 (zero-residual prior) | 2.499 mm | 0% by strict tie | 0.584 mm |
| 1,000 | 2.510 mm | 99.9% | 0.573 mm |
| 3,000 | **2.478 mm** | 99.7% | **0.567 mm** |
| 5,000 | 2.494 mm | 99.9% | 0.571 mm |

At batch 3,000, discrimination is 100.0% raw, 99.0% exact-black and 100.0% imperfect-brush, but the truth-blind selected improvement is only **0.021 mm**, far below the predeclared 0.4-mm gain and ≤2.0-mm absolute gates. The six-donor weak-real result stays within the 0.2-mm donor-regression safety limit (worst +0.036 mm at batch 3,000); those inherited Allen affines are not expert arbitrary-plane truth. The 050 useful gate fails at all trained checkpoints.

This is a clean separation between *recognizing a supplied correct atlas plane* and *finding one from the model's own imperfect hypotheses*. It confirms the 022/031/049 observation that many wrong planes provide plausible, internally coherent evidence. More epochs of this scorer, another scalar ranker, or reporting the injected-plane discrimination as localization success would be misleading. Retain 019 batch 18,000 as the internal reference. The next architecture change must improve truth-blind near-truth plane **proposal capture** across arbitrary orientation and placement; only then should whole-plane fitting feedback be reconnected to the coordinate head. No 050 checkpoint is qualified for the GUI, uncertainty calibration or DeepSlice comparison.

Frozen training: `I:/AnatomyTracker/runs/whole_plane_evidence_050_pilot`. Frozen development evaluation: `I:/AnatomyTracker/runs/whole_plane_evidence_050_development_eval` (964 rows: 177 synthetic and 64 weak-real sections at each of four checkpoints). These synthetic identities are not independent biological animals.
