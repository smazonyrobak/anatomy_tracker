# v3 one-pass pose-capture pilot 001: development result

## Decision

Update 2,000 is the **only checkpoint that passes the frozen 103 development gate**. Freeze it for a new identity-disjoint synthetic confirmation test; do not promote it to the GUI. Updates 4,000 and 8,000 gain more synthetic candidate coverage but fail the weak-real-donor guardrail, and update 4,000 also worsens the exact-black selected group. This pilot adapts candidate capture, not the requested fitting-to-coordinate feedback loop. No acquired arbitrary-plane truth, calibrated uncertainty, electrode-region probability or public DeepSlice comparison is claimed.

## Matched readout

All synthetic entries are plan-equal mean physical CCF errors in mm on the same 237 eligible v3 sections from eight 103 DEV virtual-subject plans; 176 are raw/no-brush. Each checkpoint generated its own natural 14-branch beam, which the unchanged 094 fitter and score evaluated on the same 1,024 surviving tissue pixels. “Best of 14” uses ground truth to diagnose capture and **cannot** be selected at inference. The real column is agreement with an inherited weak Allen affine, averaged equally over six separate DEV donors; it is not expert-anatomy error.

| Update | All selected | All best of 14 | Raw selected | Raw best of 14 | Weak-real donor-equal |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (exact 094 parent) | 4.156 | 2.271 | 4.612 | 2.742 | 1.072 |
| **2,000** | **3.942** | **1.900** | **4.397** | **2.217** | **0.799** |
| 4,000 | 3.288 | 1.265 | 3.492 | 1.356 | 1.605 |
| 8,000 | 3.123 | 1.138 | 3.251 | 1.208 | 1.339 |

At 2,000, the preregistered raw best-of-14 gain is 0.525 mm, 7/8 plans improve, overall selected error falls 0.215 mm, each exact-black/imperfect-brush and AP/DV/ML selected group stays within the 0.20-mm nonregression margin, and all six weak-real donor means stay within that margin. The later checkpoints each improve raw best-of-14 in 8/8 plans, but fail their full gates. At 4,000, the exact-black selected mean rises from 2.500 to 2.718 mm and four of six weak-real donor means worsen by more than 0.20 mm; at 8,000 three donor means do so. Their lower synthetic error is therefore not an acceptable advancement under the frozen rule.

The result also exposes an unsolved **selection gap**: even at the passing update, raw score-selected error is 4.397 mm versus 2.217 mm for the physically best beam member. Overall selection regret actually rises from 1.886 to 2.042 mm. A repeatable improvement in *available* candidates would not by itself make a working aligner; the model must learn to use anatomical fit to select and revise its plane and local warp. A single atlas render and an automatically selected 4-mm-error plane are not a high-quality GUI replacement.

A post-hoc raw-candidate audit locates the bottleneck more specifically. Averaging sections rather than equal-weighting plans, at update 2,000 the best of all 14 is 1.899 mm, the best within the eight original-bank candidates is 2.050 mm, and the actual selection is 3.931 mm. Thus about 0.151 mm of the 2.032-mm gap comes from the extra anchor candidates; about 1.882 mm remains within the original top eight. The selected branch came from that original bank in all 237 sections. The fit-adjusted score changed the prior's choice in only 19 sections. Across candidates within each section, mean Spearman correlation with physical error is −0.031 for the final score (negative is useful), +0.048 for fit energy and +0.066 for warp cost (positive is useful): nearly no useful ordering. Atlas-available fraction and supported-bin count are likewise weak within-section signals (mean ρ=−0.078 and −0.044 with error). The median within-section prior-score range is 3.028 logits versus 0.197 for the fit penalty; merely increasing that penalty is not justified when its own ordering is weak. These are exploratory diagnostics on the reused DEV set, not another independent gate.

At the most oblique nearest-axis bin (45–54.7°, only 19 eligible sections), best-of-14 mapped error is 2.103→2.137 mm and selected error 3.766→3.725 mm. The gain is not uniform across every angle group; this small cell cannot establish a sharp oblique failure boundary or real all-angle accuracy.

## Training and interpretation

Training completed 8,000 updates on `I:`: 16,000 accepted independent synthetic physical sections from 17,623 attempts, with one randomized v3 appearance per accepted plane, plus 8,000 reserved real-TRAIN presentations. The first 2,000 updates trained pose heads only; later updates also trained shared image features at lower learning rate. The mean synthetic loss over successive 2,000-update blocks fell 10.116 → 7.966 → 7.490 → 7.409. This is optimization evidence, not localization or physical-transfer evidence. The subsequent weak-real DEV regression coincides with shared-feature unfreezing, but this run alone cannot establish its cause. The one-sided TRAIN retention term did not guarantee generalization to DEV donors.

The evaluator initially hit a scalar-indexing error before creating its output directory; the one-line correction removed an erroneous `[0]` after the per-candidate mean. The rerun exited successfully. Its step-zero model tensor values exactly match the 094 parent. The frozen run and evaluator receipts bind source, protocol, checkpoint, panel, real-data and raw-result hashes. Training receipt SHA-256: `433620ceb345ea8e4426a926ae81931597aa9446c59f4a0848e5185258e137d5`; evaluation candidate/section/real-row/summary SHA-256: `5d8ff35c5b6cdb9257b5e7f2c9186a9dd40ef4f2a0b0978a33f27f7198b79a8e`, `874b7fe31828b382c3ccd216b57931831133dc5c7acc36130f9797585a8834b3`, `2e8c48c176024ab1f08ea78390aab4b8bc2ea306447ec2505a6a8233f146581f`, `570d994513f22e520e3e2f32f77a957cfe2dac35ca0f073af31454d4ba950231`. Exact outputs: `I:/AnatomyTracker/runs/v3_one_pass_pose_capture_pilot_001` and `I:/AnatomyTracker/runs/v3_one_pass_pose_capture_pilot_001_development_eval`.

Checkpoint choice was made on the reused 103 DEV panel, whose earlier raw gap motivated this pilot. It is **not an unbiased confirmation**. The frozen next test uses new independent synthetic identities and only compares the already-chosen 2,000 checkpoint with step zero. Even a pass there would leave acquired steep-oblique truth, animal-level evaluation, realistic artifact validation, calibrated uncertainty, joint fit-feedback training, and GUI integration unresolved. Keep public benchmarking sealed until these internal gates are promising.

That predeclared fresh synthetic comparison subsequently passed; see [confirmation 001](V3_POSE_CAPTURE_CONFIRMATION_001_RESULT_20261009.md). It confirms improved candidate capture, not a deployable selector or a biological-animal result.

An independent raw-row audit reproduced all counts, hashes, metrics and gate decisions. It did not independently load step-zero tensor weights, which the frozen evaluator checked and recorded as identical to the parent.
