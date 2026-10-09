# v3 pose-capture confirmation 001: fresh synthetic result

The preselected update-2,000 checkpoint **passed the frozen synthetic confirmation gate** against its exact step-zero parent. It improves availability of a useful arbitrary-plane candidate; it does **not** make the selected plane accurate enough to deploy. The fitting-to-coordinate learning requirement is still unmet, and this panel contains synthetic descendants of one Allen atlas, not new biological animals.

Eight new deformation plans (IDs 10400–10407) produced 256 independently drawn physical sections, one appearance per plane. The panel audit found 247 eligible sections (182 raw/no-brush, 34 exact-black brush, 31 imperfect brush) and nine ineligible, with AP/DV/ML-nearest orientations all represented. All 256 image hashes, source/protocol/plan bindings, identity separation from previous TRAIN/DEV panels, and the panel receipt passed post-exit audit. The evaluator compared only the already-chosen update 2,000 with step zero; the independent verifier reproduced the raw-row metrics, counts, hashes and gate.

Plan-equal mean physical CCF errors, in mm, at the same 1,024 surviving tissue pixels:

| Section group | Eligible | Step 0 selected | Step 2,000 selected | Step 0 best of 14 | Step 2,000 best of 14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 247 | 4.015 | **3.569** | 2.330 | **1.824** |
| Raw/no brush | 182 | 4.644 | **4.049** | 2.847 | **2.117** |
| Exact-black brush | 34 | 1.995 | **1.827** | 1.031 | 1.048 |
| Imperfect brush | 31 | 2.546 | **2.545** | 0.875 | 1.011 |

The frozen raw best-of-14 gain was **0.731 mm** (required ≥0.350), and all eight plans improved. Overall selected error did not regress; exact-black, imperfect-brush and AP/DV/ML selected groups each stayed within the allowed +0.200-mm margin. At 45–54.7° from the nearest cardinal axis (34 eligible sections), selected error was 4.005→3.558 mm and best-of-14 2.537→1.875 mm. This small synthetic angle group does not validate real steep-oblique cuts.

“Best of 14” uses ground truth solely to diagnose candidate capture and cannot be selected by the deployed model. The **selection gap remains large**: overall selected-minus-best regret rose from 1.686 to 1.745 mm, and raw regret from 1.797 to 1.932 mm. The frozen score is still failing to identify the better available plane. The seven-case [actual-generator visual grid](I:/AnatomyTracker/reports/v3_pose_capture_confirmation_examples_001/actual_generator_true_and_selected_planes.png) shows true atlas planes, generated observations and selected fitted planes (displayed without local warp). It includes all nearest-axis families and a 1%-visible-tissue peripheral case where the 10.3-mm error argues for ambiguity handling. These artifacts are synthetic and remain unvalidated against the lab's acquired histology.

Scientific next step: target **observable anatomical fit and candidate selection**, with an explicit no-displacement match option and a held-out spatial-consistency signal that can beat the unchanged-plane control. Only then let fit-derived loss update the direct probabilistic pose head in the same checkpoint lineage. Keep synthetic coordinate supervision, real TRAIN donor retention and new identity-disjoint readouts; do not scale the present weak scalar fit score, calibrate its logits as probabilities, promote it to the GUI, or run the full public benchmark. Independent acquired all-angle landmark truth and animal-separated calibration/final testing remain outstanding.

Frozen output: `I:/AnatomyTracker/data/v3_pose_capture_confirmation_panel_001` and `I:/AnatomyTracker/runs/v3_pose_capture_confirmation_001_eval`. Evaluation completion SHA-256 bindings: candidate rows `a317082693181b2256fa581b704b59eb0fbfc1f8bc35194e2ced6655d4d61fed`, section rows `d1af0c24c5672d5096f0d71a62476b2df06ccaf346710bf9baa7e249a9690ec2`, summary `f3fea344118b943a583dc1aa977c007a2b8c602f94d2633d9f6b5a5fe818af59`, panel receipt `7bc111eaa4015d53fb9efcf2b83320cdf41652c680d23584919aa35111ea7d63`. Step zero was tensor-equal to the 094 parent. No external pretrained weights, legacy pseudolabels, acquired expert truth, final animals, calibration or public benchmark data entered this confirmation.
