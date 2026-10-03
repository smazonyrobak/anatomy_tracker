# 055 result: full-bank training avoids collapse but does not capture the plane

After 054's equal-score collapse, 055 froze the scratch-trained 025/030 atlas tower and its 2,061,824-key bank. Only the observed-image query stem/shared trunk was updated. The loss compared each of 16 visible-tissue patches from a fresh arbitrary-plane TRAIN section against **all** atlas keys, with training-only nearby bank positions and orientations as multiple positives and a weak frozen-query preservation term. There were 1,500 independent accepted TRAIN sections across 64 synthetic training identities, with one appearance each; no external model or DEV image entered training.

The loss remained noncollapsed and improved from mean 9.392 over the first 250 batches to 9.120 over the final 250. On the unchanged 185-section/eight-identity synthetic DEV panel, with both query parities and the same **GT-valid** point selection as 030:

| Query checkpoint | Whole-brain top-one point recall, 0.5 mm | Top-16 recall, 0.5 mm | Sections with three separated correct points |
| --- | ---: | ---: | ---: |
| Parent / batch 0 | 1.86% | 15.42% | 67.49% |
| Batch 500 | 1.83% | 17.06% | 70.94% |
| Batch 1,500 | 1.95% | **17.89%** | 72.90% |

Top-16 recall improved by 2.47 percentage points, including raw (13.95→17.12%), exact-black (18.42→19.86%) and imperfect-brush (13.81→16.09%). But the predeclared necessary gate of ≥25% top-16 and ≥3% top-one **failed**. Nearly 98% of individual points still put a wrong atlas key first. The modest top-16 gain is not adequate to assemble a reliable arbitrary-plane pose, and the protocol does not support scaling this exact query-only recipe further yet. The true-alignment geometry and dense global lookalikes need further diagnosis; another small scalar scorer would not address this failure.

The run/evaluator exited normally. Read-back matched all checkpoint, training-log, draw, evaluator config/rows/summary hashes. The 1,500 accepted training physical-section IDs are unique; the 555 evaluation rows cover exactly the same 185 DEV sections at three checkpoints; TRAIN and DEV identity sets are disjoint. The parent checkpoint exactly reproduced 030's frozen baseline. These are synthetic components, not animal-level biological validation. Query locations were selected using synthetic truth, not an inference method. No final-test animal, public DeepSlice benchmark, calibrated probability or GUI replacement was used.

Frozen training: `I:/AnatomyTracker/runs/frozen_bank_global_query_055_pilot`. Frozen evaluation: `I:/AnatomyTracker/runs/frozen_bank_global_query_055_development_eval`.
