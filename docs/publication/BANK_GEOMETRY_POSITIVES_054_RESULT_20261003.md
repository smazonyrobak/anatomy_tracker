# 054 result: per-query rigid-bank contrast collapsed at chance

053 showed that 025's original positive atlas patches contained the synthetic slice's exact local warp, while deployment can search only rigid atlas patches. 054 tried to correct the training geometry directly: each new arbitrary-plane TRAIN section supplied 16 surviving tissue points; each point had four nearest rigid bank-frame positives at the nearest atlas-bank position and 16 globally mined distant negative keys. Only the observed image carried tissue deformation. The 025 scratch-trained weights and optimizer were continued; no external model was used.

The loss began at 4.845 but reached **1.610 at batch 500 and 1.610 at batch 1,000**, with tiny gradients (0.014 and 0.033). For four positives and 16 negatives, a representation assigning equal scores to all candidates has loss `log(20/4) = 1.609`. The mean loss of the final 200 processed batches was 1.6096. This is a chance-level representation, not successful fitting. The trainer was deliberately stopped after its complete batch-1,000 checkpoint rather than spending the remaining budget on the collapsed pilot; its partial log contains 1,200 completed batches. No `completed.json` exists for the planned 5,000-batch run. The incomplete run is not presented as a finished 5,000-batch experiment.

The frozen batch-0 and batch-1,000 checkpoints were evaluated on the same 185 paired DEV sections and **correct rigid true-plane** comparison as 053. Batch 0 exactly reproduced the 053 rigid baseline:

| Identity-equal rigid local metric | Batch 0 | Batch 1,000 |
| --- | ---: | ---: |
| Top-one point recall within 0.5 mm | 75.20% | **44.12%** |
| Top-16 point recall within 0.5 mm | 97.42% | 77.96% |
| Exact-pixel candidate first | 59.67% | 30.45% |
| Raw / exact-black / imperfect-brush top-one | 73.35% / 80.78% / 72.88% | 21.01% / 66.12% / 43.13% |

The predeclared small-bank gate fails severely. No rebuilt whole-brain bank is warranted. This does **not** invalidate the measured train/search geometry mismatch from 053; it shows that the chosen isolated four-positive/16-negative objective is unstable when the positives are harder. The objective did not require a query to discriminate among all bank locations, and its equal-score solution proved attractive in practice. A later attempt must preserve a noncollapsed atlas representation and optimize the actual many-way retrieval problem before another large bank rebuild.

The early-stopped trainer and evaluator outputs are on `I:/AnatomyTracker/runs/bank_geometry_positives_054_pilot` and `I:/AnatomyTracker/runs/bank_geometry_positives_054_early_stopped_rigid_local_eval`. Independent read-back matched the evaluation config/rows/summary and both checkpoint and partial-log hashes. There are exactly 370 evaluation rows (185 sections × two checkpoints), eight DEV synthetic identities, and disjoint TRAIN synthetic animal IDs. The partial draw log has 1,216 accepted distinct physical-section IDs, of which 1,200 have completed training rows; batch 1,000 is the only post-start checkpoint. No images were visually inspected. No final-test animals, public benchmark, probability calibration or GUI model replacement was used.
