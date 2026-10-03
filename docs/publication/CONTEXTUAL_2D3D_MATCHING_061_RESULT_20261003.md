# 061 contextual matching: frozen development result

**Decision: do not advance 061.** Candidate-conditioned 2D/3D context exchange did not pass the [predeclared development gate](CONTEXTUAL_2D3D_MATCHING_061_PROTOCOL_20261003.md). It must not replace the parent, enter the GUI, or be used to support calibrated probabilities. This was a matcher-head pilot on a frozen 059 backbone, not a trained end-to-end pose/deformation model.

## Frozen run and data

`training/train_pose_feedback_contextual_061.py` ran 6,000 batches: 12,000 independently drawn synthetic TRAIN sections and 6,000 weak-affine real TRAIN draws. The train completion receipt records draw SHA-256 `4cd7155d5b377d1e913d626612e4685959546e8c38807bee6a25919187421e51`, training SHA-256 `5c5748ce1fb5aae2c7cdd18d75bf7e63c9388c592606be0a45284ebed9f640b8`, and config SHA-256 `1921599dfb0ff9461d05618ad23228823867cf6efb95ee3c93eddfe8458c6ae0`. Parent backbone weights remained frozen; the 041 head initialized the contextual head. No legacy application weights, external pretrained features, public benchmark labels, or true-plane injection at inference were used.

`training/prepare_pose_feedback_061_fresh_dev_plans.py` and `training/prepare_pose_feedback_061_fresh_dev_panel.py` made a new, identity-disjoint synthetic DEV panel: 8 synthetic deformation identities, 32 independently sampled sections each, 246/256 eligible sections. It contains 82 raw, 104 exact-black, and 70 imperfect-brush appearances. Those are independently drawn planes, not deliberate paired-background copies. The plans receipt SHA-256 is `e79f92140d53120063ffeaec51cbe97320a197c23a146a6decce3a6355487450`; panel records SHA-256 is `481a66fd4697e5ee6de737787341ddb426b0b8655c9bc9c02d1234522a3d766a`. Animal/specimen/experiment/section identifiers and source hashes remain in each record. These synthetic identities are **not biological animals**.

`training/evaluate_pose_feedback_contextual_061.py` evaluated checkpoints 0/1,000/3,000/6,000 on those 246 eligible synthetic sections and 64 separate weak-affine real DEV sections from 6 donors. Its 1,240-row receipt has rows SHA-256 `5c8bc9f7f32137b4bd9d4d8d81b2e6bb1ab2d66d5a95c278ac9eded730ab7c71`, summary SHA-256 `53a0db63533c21458d63e6cf64eb7ab80bde5a6c9d64cdb6e06dae8310132865`, and config SHA-256 `ebdc77ca344581a69a7712d5c549eccf07f470da62376422ebfe49b90aa5f7dd`. The frozen source/checkpoint/panel hashes and all receipt hashes were checked after process exit. Raw results and receipts are on `I:` only.

## Result

Primary synthetic figures below are the frozen identity-equal means in mm; lower is better. The matched parent is constant across checkpoints. “Best of 14” chooses by synthetic truth among the actual proposed candidates and measures *candidate availability*, not a deployable selector.

| Checkpoint | Parent selected rigid | Fitted selected rigid | Parent best of 14 | Fitted best of 14 | Fitted selected mapped | Weak-real fitted selected |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.656 | 2.588 | 1.022 | 1.018 | 2.583 | 0.723 |
| 1,000 | 2.656 | 2.693 | 1.022 | 1.092 | 2.689 | 0.685 |
| 3,000 | 2.656 | 2.581 | 1.022 | 1.117 | 2.574 | 0.680 |
| 6,000 | 2.656 | 2.626 | 1.022 | 1.140 | 2.624 | 0.661 |

At 6,000, selected rigid error improved only **0.029 mm** against the matched parent (required at least 0.25 mm), while best-of-14 worsened **0.118 mm** (required at least 0.20 mm improvement). The 3,000 checkpoint similarly missed both criteria. All four checkpoints failed the frozen gate. The worst real-donor weak-reference regression was 0.138 mm at 6,000, within the 0.20-mm guardrail, but the real metric is only agreement with an inherited weak Allen affine, not known correct arbitrary-plane registration.

Fine-scale correspondence also deteriorated on unseen synthetic identities. Selected-candidate matching within 0.5/1.5 mm fell from 5.0%/44.6% at checkpoint 0 to 2.8%/35.1% at 6,000. On the physically best prior candidate, 0.5/1.5-mm recall fell from 10.6%/79.3% to 5.6%/60.8%, although the correct key was available within 0.5/1.5 mm for 29.2%/100% of those queries. An atlas-contrast ablation raised final selected error from 2.626 to 2.861 mm; the model uses atlas intensity, but that dependency is not evidence of accurate matching.

Visibility and appearance matter. For the exploratory per-section visibility bands (<10%, 10–25%, >=25% visible tissue; n=56/143/47), parent selected errors were 3.749/2.352/2.281 mm, versus 061 final 3.859/2.319/2.115 mm. Thus the small gain is confined to better-covered sections; sparse sections regress. Final fitted selected error was 2.496 mm for exact black, 2.555 for imperfect brush, and 2.869 for raw backgrounds, versus parent 2.508/2.665/2.843 mm. These bands are section-weighted diagnostics, not the identity-equal primary estimate.

The narrower conclusion is that this contextual head, training schedule, and frozen backbone did **not** improve generalizable full-frame registration. The drop in holdout point recall despite lower training correspondence loss is consistent with a generalization or supervision mismatch; it does not identify its sole cause. Training longer on the same loss or adding another scalar selector is not justified by this result. Keep the parent as research reference and investigate the missing information/supervision needed for globally correct arbitrary-plane poses before expensive joint training. No public DeepSlice benchmark, untouched animal final test, expert truth set, calibration, or GUI replacement was performed.
