# 170: anatomy-specific fitter fails blind plane selection

The [frozen protocol](ANATOMY_CORRESPONDENCE_170_BLIND_TRAIN_PROTOCOL_20261011.md) evaluated all five 169 checkpoints on one fresh 64-attempt synthetic TRAIN panel. Fifty-eight attempts had usable tissue. Each got the same blind 16-candidate beam from the frozen direct model; no plane truth entered proposal generation, scoring or fitting. Post-exit audit matched every output hash in the completion receipt, all 169 run and source hashes, and the exact 132 parent. The first evaluator attempt (`anatomy_correspondence_170_blind_train_001`) stopped before scoring because it compared a checkpoint's Python tuple with an equivalent JSON list; the corrected evaluator wrote a new `..._002` directory. No scientific row from `_001` entered this result.

| 169 batch | Direct prior | Native 121 fitter | New anatomy score after re-render | Atlas-intensity-null score | Best of 16, truth-chosen |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.516 | 2.514 | 3.232 | 3.541 | 1.116 |
| 1,500 | 2.516 | 2.514 | 3.098 | 3.177 | 1.116 |
| 3,000 | 2.516 | 2.514 | 3.211 | 3.260 | 1.116 |
| 4,500 | 2.516 | 2.514 | 3.115 | 3.224 | 1.116 |
| 6,000 | 2.516 | 2.514 | **2.922** | 3.134 | 1.116 |

Values are mean **initial-candidate** rigid-gauge errors in mm over observed tissue, not final anatomical/electrode errors. At batch 6,000, the direct prior captured a candidate within 1.5 mm in 19/58 cases, the native fitter in 20/58, the anatomy score in only 16/58, and a truth-chosen member of the blind beam in 46/58. Anatomy ranking improved 24 cases over the prior, worsened 26 and tied eight; several wrong choices added 3–5 mm. The selected anatomy candidate's fitted error was 2.912 mm, only 0.010 mm lower than its initial error, while native-fitted error was 2.514 mm. Thus neither candidate selection nor its local correction overcame the gap to available good candidates.

The score used atlas intensity: by the final checkpoint it beat its intensity-null re-rendered selection by 0.212 mm. That is **not** enough to beat either practical baseline. On the 39 raw-background cases anatomy was 3.212 mm versus prior 2.508 and intensity-null 3.190 mm; on 11 exact-black cases anatomy was 1.871 versus prior 2.348 mm. These small, unequal strata are diagnostic, not proof of an acquisition-mode effect. Anatomy was also poor for six cases under 10% tissue support (4.194 versus prior 3.181 mm) and five heavily oblique cases (3.264 versus prior 2.295 mm); do not extrapolate those tiny groups to physical cuts.

**Decision: fail the feedback-signal gate.** The 169 same-support 85.7% contrast decision and 0.700→0.631-mm near correction did not transfer to choosing among the model's own imperfect planes. Do not connect this score to the direct pose-head gradient, extend its training unchanged, promote it to real DEV/public benchmark or install it in the GUI. The immediate targeted change is to train/evaluate on blind model-generated hard candidate beams with image-only anatomical matching and a stronger spatial/contextual descriptor, retaining matched atlas-intensity-null and prior/native controls. A small independent TRAIN panel must first demonstrate candidate selection, then fresh synthetic DEV and animal-separated weak-real guardrails can decide whether joint fitting-to-pose training is justified. The present study has no physical steep-oblique truth, calibrated uncertainty or electrode-region probability claim.

Frozen scientific output: `I:/AnatomyTracker/runs/anatomy_correspondence_170_blind_train_002`; completion SHA-256 `acdc51c320208d4445cd8088dc47fce3647f44c6449a2baa6d632c99b2aac812` binds configuration, all case rows and stratified summaries.
